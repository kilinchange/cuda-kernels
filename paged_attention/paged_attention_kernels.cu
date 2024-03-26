/**
 * 模板参数信息：
 * scalar_t: query、key 和 value 的数据类型，例如 FP16
 * HEAD_SIZE：每个 head 的元素个数（即一个 token 有多少个元素 TODO:?)
 * BLOCK_SIZE：每个 block 中的 token 个数
 * NUM_THREADS：每个 thread block 中的 thread 数目
 * PARTITION_SIZE：张量并行 GPU 的数量（简单起见，假设该值为 0，且张量并行被禁用）
 *
 * 一些概念：
 * Sequence：序列，表示一条客户请求。例如，q 指向的数据形状为 [num_seqs, num_heads, head_size]，这代表 q 指向了一共 num_seqs 条查询序列。
 *           由于该 kernel 是 single query attention kernel，且每个序列只有一个 query token（即 decoding 阶段）。
 *           因此，num_seqs 等于 batch 里处理的所有 token 数目（即复用 num_seqs 维度来表示 batch）。
 * Context：上下文，由序列中生成的 token 组成。例如，["what", "is", "your"] 是上下文 tokens，而输入的 query token 是 "name"。
 *          模型输出 token 可能是 "?"。
 * Vec：vec 是一组被同时取用和计算的元素（向量化优化）。对于 query 和 key 数据，VEC_SIZE 值确保了每个 thread group 一次可以取用和计算 16 字节的数据。
 *      对于 value 数据，V_VEC_SIZE 值确保了每个 thread 一次可以取用和计算 16 字节的数据。例如，如果 scalar_t 是 FP16（2 bytes），且 THREAD_GROUP_SIZE=2，那么 VEC_SIZE=4, V_VEC_SIZE=8。
 * Thread group：线程组，是一小组线程（大小为 THREAD_GROUP_SIZE），一次获取并计算一个 query token 和一个 key token，每个线程仅处理 token 的一部分，
 *               一个线程组处理的元素总数是 x（TODO:?）。例如，如果线程组包含 2 个线程，head_size 是 8，那么 0 号线程将处理索引为 0,2,4,6 处的 query 和 key 元素，
 *               而 1 号线程处理索引 1,3,5,7 处的元素。
 * Block：指 key cache 和 value cache 块，每个块在一个 head 里存储固定数量（BLOCK_SIZE）的 token 数据，每个块可能仅包含整个上下文 token 的一部分。
 *        例如，如果 block_size=16，head_size=128，则对于 1 个 head，1 个 block 可以存储 16 * 128 = 2048 个元素。
 * Warp：warp 是一组在 SM 上同时执行的 32（WARP_SIZE） 个线程。在此 kernel 中，每个 warp 一次处理一整个 block 中的 query token 和 key token 的计算（可能在多次迭代中处理多个 block）。
 *       例如，如果一个上下文有 4 个 warp 和 6 个 block，则将会进行类似分配：warp0 处理第 0 个 block 和第 4 个 block，warp1 处理第 1 个 block，
 *       warp2 处理第 2 个 block，warp3 处理第 3 个 block。
 * Thread block：即 cuda 里的 block，是一组可以访问同一共享内存的线程（NUM_THREADS）。每个线程块包含多个 warp（NUM_WARPS）。
 *               在这个 kernel 中，每个线程块处理一个 query token 和整个上下文的 key tokens 之间的计算。
 * Grid：cuda 里的概念，是线程块的集合。在该 kernel 中，grid 的形状是 [num_heads, num_seqs, max_num_partitions]。因此，每个线程块只处理一个头、一个序列、一个分区的计算。
 */
template <
    typename scalar_t,
    int HEAD_SIZE,
    int BLOCK_SIZE,
    int NUM_THREADS,
    int PARTITION_SIZE = 0>
__device__ void paged_attention_kernel(
    ...                                    // Other side args.
    const scalar_t* __restrict__ out,      // [num_seqs, num_heads, max_num_partitions, head_size]
    const scalar_t* __restrict__ q,        // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads, head_size, block_size]
    ...                                    // Other side args.
) {
   // load Q
   // 每个线程组获取一个 query token，而每个线程本身仅处理一个 query token 数据的一部分。在每个 warp 中，每个线程组将获取相同的 query token 数据，但会将其与不同的 key token 数据相乘。
   // 每个线程定义自己的 q_ptr，指向全局内存中的 query token data。例如，如果 VEC_SIZE 为 4，HEAD_SIZE 为 128，则 q_ptr 指向包含总共 128 个元素的数据，分为了 128/4=32 个 vec。
   const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
   // 接下来，需要将 q_ptr 指向的全局内存数据读取到共享内存中作为 q_vecs，为了利用内存合并，vec 按行分配。
   __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];

   // load K
   // 与 q_ptr 不同，每个线程中的 k_ptr 会在不同的迭代中指向不同的 key token。
   const scalar_t* k_ptr = k_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride + physical_block_offset * x;
   // 接下来，需要从 k_ptr 读取 key token 数据并将它们作为 k_vecs 存储在寄存器内存中。
   // 对 k_vecs 使用寄存器内存，因为它只会被一个线程访问一次，而 q_vecs 将被多个线程访问多次。
   K_vec k_vecs[NUM_VECS_PER_THREAD];

   // QK
   // 在整个 for 循环块之前，我们获取一个 token 的查询数据并将其存储在 q_vecs 中。
   // 然后，在外部 for 循环中，我们迭代指向不同 token 的不同 k_ptr，并在内部 for 循环中准备 k_vec。
   // 最后，我们在 q_vecs 和每个 k_vecs 之间执行点乘。
   // 如前所述，对于每个线程，它一次仅获取部分 query 和 key token 数据。
   // 但是 QK_dot<>::dot 会发生跨线程组的规约，因此 qk 实际得到的是整个 query 和 key token 数据之间点乘的完整结果。
    q_vecs = ...
    for ... {
        k_ptr = ...
        for ... {
            k_vecs[i] = ...
        }
        ...
        float qk = scale * QK_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
    }

    // Softmax
    // 接下来，我们需要计算所有 qk 的归一化 softmax。需要计算点积最大值 qk_max 和 softmax 分母里的幂次和 exp_sum。
    // 上述两个对点乘 qk 的规约操作应当在整个线程块进行，即包括 query token 和所有上下文 key token 之间的结果。
    // -- qk_max 和 logits
    // 得到 qk 结果后，我们可以用 qk 设置临时的 logits 结果（最后 logits 应当存储归一化的 softmax 结果 TODO:?），我们还可以比较并收集当前线程组计算的所有 qk 的 qk_max。
    // 请注意，这里的 logits 位于共享内存上，因此每个线程组将为自己分配的上下文标记设置字段。总的来说，logits 的大小应该是上下文标记的数量。
    if (thread_group_offset == 0) {
       const bool mask = token_idx >= context_len;
       logits[token_idx - start_token_idx] = mask ? 0.f : qk;
       qk_max = mask ? qk_max : fmaxf(qk_max, qk);
    }
    // warp 规约。然后我们需要得到每个 warp 里所有线程的规约结果 q_max。主要的思路是让 warp 中的线程互相通信，并得到最终的最大 qk 值。
    for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
       qk_max = fmax(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
    }
    if (lane == 0) {
       red_smem[warp_idx] = qk_max;
    }
    // 线程块规约。最后，我们可以通过比较该线程块中所有 warp 的 qk_max 得到最终的规约结果，然后将结果广播给每一个线程。
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
       qk_max = fmax(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
    }
    qk_max = VLLM_SHFL_SYNC(qk_max, 0);

    // -- exp_sum
    // 与 qk_max 类似，我们也需要从整个线程块中获取 exp 的规约求和值。
    // 首先，对每个线程组的所有 exp 值求和，同时将 logits 的每个元素从 qk 转换为 exp(qk - qk_max)。然后我们可以像 qk_max 一样在整个线程块上规约 exp_sum。
    for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
       float val = __expf(logits[i] - qk_max);
       logits[i] = val;
       exp_sum += val;
    }
    ...
    exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);
    // 最后，通过规约的 qk_max 和 exp_sum，我们可以得到最终的归一化的 softmax 结果并存储在 logits 里。
    // 该 logits 变量将在后续的步骤中与 value 数据进行点乘。现在，它存储的是所有上下文 token 的归一化 softmax 结果。
    const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
    for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
       logits[i] *= inv_sum;
    }

    // Value
    // 现在我们需要读取 value 数据，并与 logits 执行点乘操作。与 query 和 key 不同，处理 value 数据时没有线程组的概念。
    // 首先简单介绍 value 数据的内存排布：一个 value block 有 HEAD_SIZE 行和 BLOCK_SIZE 列，且被分割成多个 v_vec，同一列的的数据对应相同的 value token。
    // 每个线程一次从 V_VEC_SIZE 个 token 中各取（相同索引处的）一个元素，共得到 V_VEC_SIZE 个元素作为 v_vec。对于每个 v_vec，需要与对应的 logits_vec 进行点乘（logits_vec 是来自 logits 的 V_VEC_SIZE 个元素）。
    // 于是有了下面两层迭代，通过多次内部迭代，每个 warp 将处理完一个 block 的 value tokens；并且通过多次外部迭代，完成整个上下文的 value token 的处理。
    // 具体而言，在外层循环中，logits_vec 遍历不同的 block，并从 logits 里读取 V_VEC_SIZE 个元素；
    // 在内层循环中，每个线程从 V_VEC_SIZE 个 tokens 里读取 V_VEC_SIZE 个元素到 v_vec 中，并与 logits_vec 执行点乘。
    // 注意，这里每个线程拿到的是 tokens 的不同索引处的元素，点乘结果被累加到 accs 中。(TODO:?)
    float accs[NUM_ROWS_PER_THREAD];
    for ... { // Iteration over different blocks.
        logits_vec = ...
        for ... { // Iteration over different rows.
            v_vec = ...
            ...
            accs[i] += dot(logits_vec, v_vec);
        }
    }

    // LV
    // 现在，我们需要在每个 warp 内对 accs 进行规约，让每个线程得到一个 block 中 token 指定位置的 accs 累计值。(TODO:?)
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      float acc = accs[i];
      for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
        acc += VLLM_SHFL_XOR_SYNC(acc, mask);
      }
      accs[i] = acc;
    }
    // 接下来，我们对所有 warp 进行规约，让每个线程都获取所有上下文 token 的指定位置的 accs 累计值。(TODO:?)
    float* out_smem = reinterpret_cast<float*>(shared_mem);
    for (int i = NUM_WARPS; i > 1; i /= 2) {
      // Upper warps write to shared memory.
      ...
        float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
          ...
          dst[row_idx] = accs[i];
        }

        // Lower warps update the output.
          const float* src = &out_smem[warp_idx * HEAD_SIZE];
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
                    ...
          accs[i] += src[row_idx];
        }
            // Write out the accs.
    }

    // Output
    // 现在将所有计算结果从本地寄存器内存写入最终要输出的全局内存中。
    // 首先，定义 out_ptr 变量，指向指定序列和头的起始地址。
    scalar_t* out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                    + head_idx * max_num_partitions * HEAD_SIZE
                    + partition_idx * HEAD_SIZE;
    // 接下来，不断遍历不同指定头部位置，并根据 out_ptr 写出相应的累计结果。(TODO:?)
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
}
