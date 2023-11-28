### 基于CFG 语法模板的文本提取


### 先决条件

- 需要先下载好开源模型，比如baichuan2, chatglm2, llama2,Mistral 
- 本代码库的作用是在您的开源模型的基础上，通过使用CFG模板来限制模型输出，以得到理想的输出格式


####  原理：
- 模型是一个token 一个token地去生成结果 
- 模型每生成完一个token,我们便根据cfg语法和当前已经生成的文本去判断接下来的token的取值范围
- 然后让模型从这个范围内选择概率最高的那个token (也就是torch.argmax(logits)这个操作)，作为新生成的token
- 不断循环上述过程，直到生成一个完整的json文档
- 本项目中我们先确定好json文件的格式，以及对应的key，然后让模型根据这些key去生成回答


### CFG 语法

- CFG 的由non-terminal 和 terminal组成，它们是什么？参考如下例子:

```
non_terminal_1 : "[" [non_terminal_2  ["," non_terminal_2]*] "]"  % 规定结果里要有最少一个non_terminal_2
non_terminal_2 : "{" non_terminal_3 "," non_terminal_4  "}"  % 规定non_terminal_2是由 non_terminal_3和non_terminal_4组成
non_terminal_3 :  terminal_1   ":"  terminal_2 % 规定 non_terminal_3是 terminal_1 和 terminal_2组成的键值对
non_terminal_4 : terminal_3   ":" terminal_4 % 规定 non_terminal_4 是 terminal_3和 terminal_4组成的键值对
terminal_1 : /"key1"/  % 规定terminal_1是这样一个正则表达式 re.compile('"key1"')
terminal_2 : /"val1"/  % 规定terminal_2是这样一个正则表达式 re.compile('"val1"')
terminal_3 : /"key2"/ % 规定terminal_3是这样一个正则表达式 re.compile('"key2"')
terminal_4 : /"val2"/ % 规定terminal_4是这样一个正则表达式 re.compile('"val2"')
```
- 上面例子中/ blah blah blah ... /中的内容是正则表达式
- 所以terminal 是正则表达式, non-terminal是正则表达式组成的规则


### TODO List

- [ ] batch processsing 批处理
- [ ] 对CFG 的get_accept_token_ids 函数进行多线程处理
- [ ] 完善对主流模型的支持