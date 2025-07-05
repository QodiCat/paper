# PEFT方法类详解

## 1. 核心思想

PEFT（Parameter-Efficient Fine-Tuning）是一种基于**参数高效微调**的持续学习方法。它为每个任务学习独立的适配器（如LoRA、PromptTuning），同时保持主模型参数不变，从而避免灾难性遗忘。

## 2. 支持的PEFT类型

### 2.1 基础PEFT类型
```python
parser.add_argument("--PEFT_type", type=str, default='PromptTuning', 
                   choices=['PromptTuning','LoRA'], help="The peft type")
parser.add_argument("--Sub_PEFT_type", type=str, default='NewLoRA', 
                   choices=['NewLoRA'], help="The peft type")
```

### 2.2 PromptTuning参数
```python
parser.add_argument("--PEFT_num_virtual_tokens", type=int, default=10, 
                   help="The number of tokens for prompt tuning")
parser.add_argument("--PEFT_prompt_tuning_init_text", type=str, default='auto', 
                   help="The initialization words for prompt tuning")
```

### 2.3 LoRA参数
```python
parser.add_argument("--PEFT_lora_r", type=int, default=4, help="The rank of lora")
parser.add_argument("--PEFT_lora_alpha", type=int, default=8, help="The scaling of lora")
parser.add_argument("--PEFT_lora_bias", type=str, default="none", help="The bias of lora")
parser.add_argument("--PEFT_lora_dropout", type=float, default=0.1, help="The dropout rate of lora")
```

## 3. 核心组件

### 3.1 适配器管理
```python
def build_backbone(self):
    self.model, self.tokenizer = get_backbone(self.params, num_task=self.CL_dataset.continual_config['NUM_TASK'])
    num_task = self.CL_dataset.continual_config['NUM_TASK']
    
    # 为每个任务创建独立的适配器
    for t_id in range(num_task):
        peft_config = self.model.peft_config['default']
        self.model.add_adapter(adapter_name='task-%d'%(t_id), peft_config=peft_config)
```

### 3.2 NewLoRA的权重选择网络
```python
# 动态适配器权重选择网络
self.mlp = nn.Sequential(
    nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2),
    nn.ReLU(),
    nn.Linear(self.model.config.hidden_size//2, num_task),
    nn.Sigmoid()  # 输出每个任务的权重
)
```

## 4. 任务级别操作

### 4.1 任务开始
```python
def begin_task(self, task_id):
    super().begin_task(task_id)
    # 设置当前任务的适配器
    self.model.set_adapter('task-%d'%(task_id))
```

### 4.2 任务结束
```python
def end_task(self, task_id):
    super().end_task(task_id)
    # 保存LoRA模型
    if self.params.PEFT_type == 'LoRA':
        if self.accelerator.is_main_process:
            save_path = os.path.join(self.params.dump_path, f'lora_task_{task_id}')
            self.model.save_pretrained(save_path)
```

## 5. 核心训练逻辑

### 5.1 生成式模型训练
```python
if self.params.classifier == 'None':
    # 使用因果语言建模损失
    total_loss = model(**{'input_ids': lm_input['input_ids_with_ans'], 
                         'attention_mask': lm_input['attention_mask_with_ans'],
                         'labels': lm_input['labels_with_ans']}).loss
```

### 5.2 判别式模型训练（标准PEFT）
```python
elif self.params.classifier in ['Linear','CosineLinear'] and self.params.Sub_PEFT_type != 'NewLoRA':
    # 提取特征
    extracted_feature = obtain_features(params=self.params, 
                                       model=model, 
                                       lm_input=lm_input, 
                                       tokenizer=self.tokenizer)
    
    # 使用当前任务的分类器
    logits = self.classifier_list[task_id](extracted_feature)
    
    # 处理不同的分类类型
    if self.params.classification_type == 'sentence-level':
        label_idx = lm_input['label_idx_til']  # 使用TIL标签
    elif self.params.classification_type == 'word-level':
        # 复杂的word-level处理逻辑
        # ...
```

### 5.3 NewLoRA训练
```python
elif self.params.Sub_PEFT_type == 'NewLoRA':
    # 提取特征
    extracted_feature = obtain_features(params=self.params, 
                                       model=model, 
                                       lm_input=lm_input, 
                                       tokenizer=self.tokenizer)
    
    # 计算LoRA权重
    lora_weights = self.mlp(extracted_feature)
    lora_weights = lora_weights.reshape(lora_weights.shape[0], -1)
    
    # 偏好损失：鼓励为当前任务分配更高权重
    target_mask = torch.zeros_like(lora_weights)
    target_mask[:, task_id] = 1
    preference_loss = F.mse_loss(lora_weights, target_mask)
    
    # 分类损失
    logits = self.classifier_list[task_id](extracted_feature)
    logits = logits.reshape(-1, logits.shape[-1])
    label_idx = lm_input['label_idx_cil'].reshape(-1)
    label_idx = label_idx - self.CL_dataset.continual_config['PRE_ACCUM_NUM_CLASS'][task_id]
    
    # 总损失
    total_loss = self.ce_loss(logits, label_idx) + preference_loss
```

## 6. Word-Level分类的特殊处理

### 6.1 PromptTuning的token移除
```python
# 移除prompt token
if self.params.PEFT_type == 'PromptTuning':
    logits = logits[:, self.params.PEFT_num_virtual_tokens:, :]
```

### 6.2 "O"类别的特殊处理
```python
# 对于非第一个任务，需要计算"O"类别的logits
if task_id > 0:
    with torch.no_grad():
        model.set_adapter('task-%d'%(0))  # 使用第一个任务的适配器
        extracted_feature = obtain_features(...)
        model.set_adapter('task-%d'%(task_id))  # 切换回当前任务
        logits_first_task = self.classifier_list[0](extracted_feature)
        logits_O = logits_first_task[:, :, 0:1]  # 提取"O"类别
    logits = torch.cat((logits_O, logits), dim=-1)
```

### 6.3 标签映射和过滤
```python
# 过滤未见过的标签
label_idx = label_idx.masked_fill(label_idx >= self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][task_id], 0)
label_idx = label_idx.masked_fill(torch.logical_and(label_idx > 0, label_idx < self.CL_dataset.continual_config['PRE_ACCUM_NUM_CLASS'][task_id]), 0)

# 转换为TIL标签
if task_id > 0:
    label_idx[label_idx > 0] = label_idx[label_idx > 0] - self.CL_dataset.continual_config['PRE_ACCUM_NUM_CLASS'][task_id] + 1
```

## 7. 评估机制

### 7.1 CIL模式评估
```python
# CIL模式下需要进行O(num_task)次前向传播
if self.params.classification_type == 'sentence-level':
    acc = evaluate_sent_level_acc_with_classifier_adapter(
        mlp=self.mlp,                    # NewLoRA的权重网络
        model=self.model,
        classifier_list=self.classifier_list,
        cur_task_id=cur_task_id if self.params.il_mode=='CIL' else eval_task_id,
        eval_data_loader=data_loader[eval_task_id],
        tokenizer=self.tokenizer,
        accelerator=self.accelerator,
        params=self.params,
        idx2label=self.CL_dataset.continual_config['idx2label']
    )
```

### 7.2 适配器选择策略
```python
# 根据IL模式选择适配器
if self.params.il_mode == 'CIL':
    self.model.set_adapter('task-%d'%(cur_task_id))  # 使用当前任务适配器
elif self.params.il_mode == 'TIL':
    self.model.set_adapter('task-%d'%(eval_task_id))  # 使用评估任务适配器
```

## 8. 关键创新点

### 8.1 参数隔离
- ✅ **主模型冻结**：主模型参数保持不变
- ✅ **任务特定适配器**：每个任务有独立的适配器
- ✅ **完全防遗忘**：不会影响旧任务的参数

### 8.2 NewLoRA的动态选择
- ✅ **权重网络**：学习每个任务的重要性权重
- ✅ **偏好损失**：引导模型为当前任务分配更高权重
- ✅ **软选择**：通过Sigmoid实现软性任务选择

### 8.3 多模态支持
- ✅ **生成式模型**：支持因果语言建模
- ✅ **判别式模型**：支持分类任务
- ✅ **句子级/词级**：支持不同粒度的分类

## 9. 优势与局限

### 9.1 优势
- ✅ **完全防遗忘**：旧任务参数不受影响
- ✅ **参数效率**：只需要少量额外参数
- ✅ **模块化设计**：易于扩展和维护
- ✅ **理论简单**：容易理解和实现

### 9.2 局限
- ❌ **推理复杂度**：CIL模式需要O(num_task)次前向传播
- ❌ **任务数量限制**：适合任务数量不多的场景
- ❌ **选择网络开销**：NewLoRA需要额外的MLP网络
- ❌ **不支持经验回放**：无法处理需要历史数据的场景

## 10. 实际应用建议

### 10.1 超参数选择
```python
# LoRA推荐设置
PEFT_lora_r = 4           # 低秩近似的秩
PEFT_lora_alpha = 8       # 缩放因子
PEFT_lora_dropout = 0.1   # 防止过拟合

# PromptTuning推荐设置
PEFT_num_virtual_tokens = 10  # 虚拟token数量
```

### 10.2 适用场景
- **推荐**：任务数量有限、参数效率要求高的场景
- **不推荐**：任务数量很多、需要复杂任务交互的场景

## 11. 总结

PEFT方法代表了持续学习中**参数效率**的重要方向，通过：

1. **适配器机制**：为每个任务学习独立的轻量级适配器
2. **参数隔离**：完全避免参数干扰
3. **动态选择**：NewLoRA提供了智能的任务选择机制
4. **多模态支持**：适用于各种NLP任务

该方法特别适合**资源受限**且**任务数量有限**的持续学习场景，是工业应用中的重要选择。

