## 目录

1. 项目概览
2. 环境要求
3. 目录结构概览
4. 配置管理

   * 4.1 全局配置
   * 4.2 模块配置
   * 4.3 密钥管理
5. 模块安装与配置

   * 5.1 ASR 模块（FunASR）
   * 5.2 LLM 模块（GPT-4.1）
   * 5.3 TTS 模块（CosyVoice2）
   * 5.4 端到端测试
6. 启动与验证
7. 常见问题与解决方案
8. 参考链接

---

## 1. 项目概览

SoulSpeak 是一个端到端语音交互系统，支持语音输入、智能理解、语音输出的完整流程。设计原则：

* **模块化**：ASR、LLM、TTS、交互入口各自独立，可替换
* **契约优先**：配置文件集中管理，接口定义明确
* **易扩展**：新增或替换模型时，只需调整对应配置

## 2. 环境要求

| 项目需求   | 版本 / 建议                 |
| ------ | ----------------------- |
| 操作系统   | macOS / Ubuntu / CentOS |
| Python | 3.10                    |
| 虚拟环境   | conda / venv            |
| 系统依赖   | ffmpeg ≥4.0, sox ≥14.4  |
| GPU 驱动 | Nvidia 驱动 ≥460.32       |

**检查示例**：

```bash
python --version  # 确保输出 Python 3.10.x
ffmpeg -version  
sox --version
```

---

## 3. 目录结构概览

```text
SoulSpeak/
├── soul_speak/
│   ├── asr/         # ASR 模块实现
│   ├── llm/         # LLM 模块实现
│   ├── tts/         # TTS 模块实现
│   ├── mic/         # 交互入口：run_and_speak.py
│   ├── conf/        # 配置文件目录
│   └── utils/       # 公共工具
├── requirements.txt # Python 依赖
├── install.md       # 安装说明（本文件）
├── README.md        # 项目介绍
└── docs/            # 扩展文档
```

---

## 4. 配置管理

### 4.1 全局配置（`conf/config.yaml`）

```yaml
app_name: "SoulSpeak"
environment: "development"  # 可选：production, staging
log_level: "INFO"
gpu_server_ip: "<填写 GPU 服务器 IP>"
```

### 4.2 全局变量与路径（`conf/globals.yaml`）

```yaml
model_root: "./pretrained_models"
audio_root: "./audio_data"
```

### 4.3 密钥管理（`conf/secrets.yaml`）

> **注意**：务必将 `secrets.yaml` 加入 `.gitignore`

```yaml
OPENAI_API_KEY: "<your_openai_api_key>"
FUNASR_TOKEN: "<funasr_service_token>"
```

---

## 5. 模块安装与配置

### 5.1 ASR 模块（FunASR）

1. **安装依赖**：

   ```bash
   pip install torch torchvision torchaudio  # 对应 CUDA 版本
   pip install -U funasr
   ```
2. **配置文件（`conf/asr/funasr.yaml`）**：

   ```yaml
   name: funasr
   endpoint: "http://${gpu_server_ip}:9000/asr"
   language: zh
   model_name: paraformer-zh-streaming
   websocket:
     host: ${gpu_server_ip}
     port: 8765
   audio:
     frame_duration_ms: 30
     samplerate: 16000
   ```
3. **常见问题**：

   * `WebSocketConnectionError`：检查 `gpu_server_ip` 和端口
   * `ModelNotFoundError`：确认 `model_name` 与服务器部署一致

### 5.2 LLM 模块（GPT-4.1）

1. **配置 `secrets.yaml`**：

   ```yaml
   OPENAI_API_KEY: "<your_openai_api_key>"
   ```
2. **安装 SDK**：

   ```bash
   pip install openai
   ```
3. **配置文件（`conf/llm/gpt.yaml`）**：

   ```yaml
   name: gpt-4.1
   websocket:
     host: ${gpu_server_ip}
     port: 8767
   temperature: 0.7
   max_tokens: 2048
   prompt_template: |
     # CONTEXT
     ...
   ```
4. **建议校验**：

   ```python
   import openai
   openai.api_key = "<your_openai_api_key>"
   resp = openai.Model.list()
   print([m["id"] for m in resp["data"]])  # 确认 gpt-4.1 可用
   ```

### 5.3 TTS 模块（CosyVoice2）

1. **克隆与安装**：

   ```bash
   git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
   cd CosyVoice
   pip install -r requirements.txt
   pip install Cython
   # 可选：安装 vllm 加速推理
   pip install vllm==0.9.0
   ```
2. **下载预训练模型**：

   ```bash
   mkdir -p pretrained_models
   # 从 Hugging Face 下载 CosyVoice2-0.5B 放入该目录
   ```
3. **配置文件（`conf/tts/cosy_voice2.yaml`）**：

   ```yaml
   websocket:
     host: ${gpu_server_ip}
     port: 8768
   model_path: "${model_root}/CosyVoice2-0.5B"
   vllm_enable: true
   real_sr: 24000
   voices:
     - name: susu
       sample: "${audio_root}/susu.wav"
       prompt: "爆炸豆家的智能电脑..."
   default_voice: susu
   ```

### 5.4 端到端测试

```bash
# 启动 ASR 服务 (FunASR)
# 启动 TTS 服务
cd CosyVoice
python tts_websocket.py

# 运行交互脚本
python soul_speak/mic/run_and_speak.py
```

---

## 6. 启动与验证

1. **按模块启动**：依次启动 ASR、LLM、TTS
2. **运行测试脚本**：

   ```bash
   python tests/test_end_to_end.py  # 若有测试用例
   ```
3. **日志检查**：

   * 日志路径：`./logs/{模块名}.log`
   * 开启 DEBUG 级别定位问题

---

## 7. 常见问题与解决方案

| 问题描述      | 排查建议                                           |
| --------- | ---------------------------------------------- |
| 模型加载失败    | 检查模型路径和文件权限                                    |
| 网络连接超时    | 确认 IP、端口、firewall 设置                           |
| YAML 解析错误 | 使用 `yamllint` 或 `python -c 'import yaml'` 检查格式 |

---

## 8. 参考链接

* FunASR 安装与部署： [https://github.com/modelscope/FunASR#installation](https://github.com/modelscope/FunASR#installation)
* CosyVoice2 代码仓库： [https://github.com/FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
* OpenAI API 文档： [https://platform.openai.com/docs/api-reference](https://platform.openai.com/docs/api-reference)
