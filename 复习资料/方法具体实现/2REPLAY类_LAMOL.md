2REPLAY类_LAMOL.md 是生成重现

# LAMOL方法详解

## 1. 核心思想

LAMOL（LAnguage MOdeling for Lifelong Language Learning）是一种基于**伪样本生成**的持续学习方法，通过训练生成式模型来学习问答任务，并在学习新任务前生成伪历史样本来防止遗忘。

## 2. 理论基础

### 2.1 双重训练目标
- **QA目标**: 学习问答能力
- **生成目标**: 学习生成历史任务样本的能力

### 2.2 伪样本生成
- 使用当前模型生成历史任务的伪样本
- 将伪样本与新任务数据混合训练
- 避免存储真实历史数据

## 3. 核心参数

```python
def get_LAMOL_params(parser):
    parser.add_argument("--LAMOL_lambda", type=float, default=0.25, 
                       help="The weight of the generation target in LAMOL")
    parser.add_argument("--LAMOL_gamma", type=float, default=0.20, 
                       help="The ratio of psesudo old samples w.r.t the training data of new task.")
    parser.add_argument("--LAMOL_topk", type=int, default=20, 
                       help="The top-k sampling for generating psesudo old samples.")
    parser.add_argument("--LAMOL_use_task_specific_gen_token", type=bool, default=False, 
                       help="If using task-specific generation token for generating psesudo old samples.")
```

- `LAMOL_lambda`: 生成目标权重
- `LAMOL_gamma`: 伪样本与新任务数据的比例
- `LAMOL_topk`: 生成时的top-k采样
- `LAMOL_use_task_specific_gen_token`: 是否使用任务特定的生成token

## 4. 支持的配置

```python
def __init__(self, params, CL_dataset, accelerator):
    # 严格限制支持的配置
    assert params.classifier in ['None']           # 仅支持生成式模型
    assert params.il_mode in ['IIL','CIL','TIL']   # 支持所有IL模式
    assert params.classification_type == 'sentence-level'
    assert params.backbone_type == 'generative'
    assert not params.is_replay                    # 不使用传统的经验回放
```

## 5. 核心训练逻辑

### 5.1 双重损失函数
```python
def observe_batch(self, task_id, epoch_id, lm_input):
    # 1. 问答损失
    qa_loss = model(**{
        'input_ids': lm_input['input_ids_with_ans'], 
        'attention_mask': lm_input['attention_mask_with_ans'],
        'labels': lm_input['labels_with_ans']
    }).loss

    # 2. 生成损失
    generation_loss = model(**{
        'input_ids': lm_input['input_ids_with_gen_ans'], 
        'attention_mask': lm_input['attention_mask_with_gen_ans'],
        'labels': lm_input['labels_with_gen_ans']
    }).loss

    # 3. 总损失
    total_loss = qa_loss + self.params.LAMOL_lambda * generation_loss
```

### 5.2 损失组成解释
- **QA损失**: 确保模型能正确回答问题
- **生成损失**: 训练模型生成样本的能力
- **权重平衡**: λ参数控制两个目标的重要性

## 6. 伪样本生成机制

### 6.1 主要流程
```python
def generate_pseudo_buffer_samples(self, task_id: int, num_samples: int) -> List[Dataset]:
    pseudo_dataset_list = []
    
    # 为每个历史任务生成伪样本
    for t_id in range(task_id):
        cnt_num_samples = num_samples // task_id
        
        # 选择生成token
        gen_token = '__%d__'%(t_id) if self.params.LAMOL_use_task_specific_gen_token else '__gen__'
        
        # 生成样本
        while cnt_num_samples > 0:
            # 批量生成
            generate_ids_all = model.generate(**lm_input, 
                                max_new_tokens=self.params.max_seq_length-max_input_len, 
                                pad_token_id=self.tokenizer.eos_token_id,
                                do_sample=True,
                                top_k=self.params.LAMOL_topk)
```

### 6.2 生成策略
- **生成token**: 使用特殊token触发生成
- **Top-k采样**: 控制生成的多样性
- **格式过滤**: 只保留包含"__ans__"的样本
- **批量处理**: 提高生成效率

### 6.3 样本格式处理
```python
for _one_sample in generated_samples:
    if _one_sample.count('__ans__') != 1:
        continue  # 过滤格式错误的样本
    
    _question, _answer = _one_sample.split('__ans__')
    _answer = _answer.replace(self.tokenizer.eos_token, '')
    
    # 构建伪样本
    pesudo_samples_dict['input'].append(_question)
    pesudo_samples_dict['target'].append(_answer)
```

## 7. 训练数据组织

### 7.1 第一个任务
```python
if task_id == 0:
    cur_train_loader = self.train_loader_list[task_id]
```

### 7.2 后续任务
```python
else:
    train_dataset = self.train_loader_list[task_id].dataset
    
    # 生成伪样本
    pseudo_buf_dataset_list = self.generate_pseudo_buffer_samples(
        task_id=task_id,
        num_samples=int(len(train_dataset) * self.params.LAMOL_gamma)
    )
    
    # 合并真实数据和伪样本
    cur_train_loader = DataLoader(
        ConcatDataset((train_dataset, *pseudo_buf_dataset_list)),
        batch_size=self.params.batch_size,
        shuffle=True,
        drop_last=False
    )
```

## 8. 数据预处理

### 8.1 LAMOL特殊预处理
```python
pseudo_dataset = pseudo_dataset.map(preprocess_function_train_generative_LAMOL, 
                                    batched=True, 
                                    desc='Generate pseudo samples for task %d'%(t_id+1), 
                                    batch_size=1000,
                                    fn_kwargs={
                                        'params': self.params,
                                        'tokenizer': self.tokenizer,
                                        'num_task': num_task,
                                        'task_id': t_id,
                                        'input_column': input_column,
                                        'target_column': target_column,
                                        'ans_token': ans_token,
                                        'eos_token': eos_token,
                                        'gen_token': gen_token,
                                    })
```

### 8.2 数据格式
- **QA格式**: input + "__ans__" + target
- **生成格式**: gen_token + input + "__ans__" + target

## 9. 关键创新点

### 9.1 无需外部存储
- ✅ 不需要存储历史数据
- ✅ 不需要额外的缓冲区
- ✅ 隐私友好

### 9.2 生成式回放
- ✅ 通过模型生成历史样本
- ✅ 样本数量可以灵活控制
- ✅ 避免存储开销

### 9.3 双重训练目标
- ✅ 同时学习任务和生成能力
- ✅ 自监督的历史知识保持
- ✅ 端到端的统一训练

## 10. 优势与局限

### 10.1 优势
- ✅ **隐私友好**: 不存储真实历史数据
- ✅ **存储效率**: 只需要一个模型
- ✅ **灵活性**: 可控制伪样本数量和质量
- ✅ **通用性**: 适用于各种生成式任务

### 10.2 局限
- ❌ **生成质量**: 伪样本可能不如真实数据
- ❌ **模式限制**: 只支持生成式模型
- ❌ **累积误差**: 生成质量可能逐渐下降
- ❌ **计算开销**: 生成过程需要额外计算

## 11. 实际应用建议

### 11.1 超参数调节
```python
# 典型设置
LAMOL_lambda = 0.25   # 生成损失权重
LAMOL_gamma = 0.20    # 伪样本比例
LAMOL_topk = 20       # 生成时的top-k

# 调节策略：
# - 生成质量差 → 增大λ（更多关注生成能力）
# - 新任务学习困难 → 减小λ（更多关注QA能力）
# - 遗忘严重 → 增大γ（更多伪样本）
# - 训练效率低 → 减小γ（更少伪样本）
```

### 11.2 适用场景
- **推荐**: 隐私敏感、存储受限的生成式任务
- **不推荐**: 需要精确历史信息的判别任务

## 12. 总结

LAMOL代表了持续学习中**生成式回放**的创新方向，通过训练模型的生成能力来实现历史知识的保持。虽然存在生成质量的限制，但其在隐私保护和存储效率方面的优势使其成为特定场景下的理想选择。

该方法的核心思想是"**让模型学会回忆**"，通过生成伪样本来模拟经验回放，是持续学习领域的一个重要突破。

Similar code found with 1 license type