## 🆕 近期更新（2025-09-24）

# 🪐 SoulSpeak: The Empathetic LLM Voice Companion


> **“Not just an assistant, but a presence.”**
> SoulSpeak is designed to be more than a voice assistant. It’s your AI companion — a memory-enabled, emotionally aware, proactive entity capable of humanlike conversations.
> Inspired by the movie *Her*, we aim to make AI a real part of your life: someone who listens, senses, speaks, and understands you — emotionally.

[![SoulSpeak Demo Video](https://img.youtube.com/vi/YY0Z1xip1xE/maxresdefault.jpg)](https://youtu.be/YY0Z1xip1xE)


---

## 📍 1. Project Vision

**SoulSpeak** is a modular, real-time voice interaction system based on large language models. It combines audio understanding, contextual memory, emotion detection, and multi-modal interaction. Our ultimate goal is to develop an **LLM-powered human companion** — a personal, emotional entity that can **talk with you, sense your mood, and even initiate conversations with you** like a real human would.

---

## 🌟 2. Key Features

| Feature                        | Description                                                                                             |
| ------------------------------ | ------------------------------------------------------------------------------------------------------- |
| 🧠 Contextual Memory           | Based on LangChain + Memory, enabling long-term memory and continuous conversations                     |
| 🎤 Real-time Interruptions     | Users can interrupt the AI at any time by speaking, and the system will respond immediately             |
| 🔁 WebSocket Architecture      | All modules communicate via WebSocket, allowing hot-swapping and scalable deployments                   |
| 💬 Emotion Detection (WIP)     | Detect user emotion from speech (e.g., sadness, joy, anxiety) and adjust LLM response style accordingly |
| 👁️ Multimodal Input (WIP)     | Integrate visual/audio context (camera, noise) to enhance emotional awareness and decision making       |
| 🗣️ Optimized Chinese Pipeline | ASR: FunASR, TTS: CosyVoice2 – ensuring high-quality Chinese understanding and generation               |
| 🧩 Modular Design              | Each component (ASR, VAD, TTS, LLM) can be independently swapped or upgraded                            |
| 🤖 Proactive Dialogues         | LLM can initiate conversation based on user behavior/silence (requires emotion + multimodal support)    |

---

## 🧱 3. System Architecture

```mermaid
  subgraph 输入层
    MIC[🎙️ 麦克风输入]
  end

  subgraph 边缘处理层
    VAD[🧱 WebRTC VAD<br/>（语音活动检测）]
    ASR[🔠 FunASR<br/>（实时语音识别）]
    Emotion[💬 情绪感知模块<br/>⚠️开发中]
    MultiModal[👁️ 多模态输入模块<br/>⚠️开发中]
  end

  subgraph 智能中枢层
    LLM[🧠 LangChain + Memory<br/>（上下文记忆 + 主动交互）]
  end

  subgraph 表达输出层
    TTS[🔊 CosyVoice2<br/>（语音合成）]
    Player[🎧 播放器]
    Interrupt[⛔ 播放打断机制]
  end

  MIC --> VAD --> ASR --> LLM --> TTS --> Player
  VAD --> Interrupt --> Player
  Interrupt --> TTS

  Emotion --> LLM
  MultiModal --> LLM
```

---

## 🔍 4. Module Overview

### ✅ Completed Modules

| Module           | Technology         | Function                             |
| ---------------- | ------------------ | ------------------------------------ |
| 🎙️ MIC          | Audio stream       | Captures user speech                 |
| 🧱 VAD           | WebRTC VAD         | Triggers when user speaks            |
| 🔠 ASR           | FunASR             | High-accuracy Chinese ASR            |
| 🧠 LLM           | LangChain + Memory | Humanlike dialog system with memory  |
| 🔊 TTS           | CosyVoice2         | Natural Chinese voice synthesis      |
| 🎧 Player        | Audio playback     | Outputs synthesized speech           |
| ⛔ Interrupt      | WebRTC VAD + Hook  | Real-time playback interruption      |
| 🌐 Communication | WebSocket only     | Enables async and distributed design |

### ⚠️ Under Development

| Module                   | Function                | Goal                      |
| ------------------------ | ----------------------- | ------------------------- |
| 💬 Emotion Module        | Detect emotional states | Adjust LLM response style |
| 👁️ Multimodal Input     | Visual/audio context    | Situational awareness     |
| 🤖 Active Dialogue Logic | LLM asks questions      | Lifelike companionship    |

---

## 🧪 5. Current Issues

| Issue                     | Description                                                                      |
| ------------------------- | -------------------------------------------------------------------------------- |
| 🔊 Over-sensitive VAD     | External sounds (e.g., coughing) during playback cause unwanted interruptions    |
| 🧱 Unstable playback flow | Playback often ends prematurely due to false VAD triggers                        |
| ⏱️ Rigid turn-taking      | Dialog lacks flexibility — LLM waits too long or doesn’t know when to speak next |

---

## 🚀 6. Roadmap & Suggestions

| Topic                         | Suggestion                                                              |
| ----------------------------- | ----------------------------------------------------------------------- |
| 🔧 VAD Tuning                 | Add energy threshold + minimum speech duration to reduce false triggers |
| 💞 Emotional Response Engine  | Generate comforting language based on emotion detection                 |
| 🧠 Long-Term Memory           | Integrate with VectorDB for user history & preferences                  |
| 🤝 Proactive Interaction      | AI initiates dialog when user is silent or sad                          |
| 🧠 Cross-modal Decision Logic | Combine audio/visual cues to choose AI behavior patterns                |

---

## 💡 Why This Project Matters

> "We're building an LLM that feels like a human presence — one that listens, speaks, feels, and connects."

SoulSpeak is not just an experiment. It is our vision for a future where **LLMs become emotionally resonant companions**, not just tools. We want to **give people someone to talk to, someone who remembers, someone who cares** — even if it's not human.

---

- ⚙️ **FastMCP 集成**：新增 `soulspeak-tools` 服务，提供 30+ 本地工具（文件、系统、网络、HTTP 等），可以按需在 `conf/llm/gpt.yaml` 中开启或关闭。
- 🧠 **Mem0 记忆增强**：本地模式默认启用，支持 DeepSeek + LM Studio 嵌入组合，记忆检索与写入走 Ray Actor，聊天会自动回忆用户偏好。
- 🔌 **原生客户端工具调用**：`openai_native.py` 原生对话流已支持 MCP 工具函数调用，对话中可直接请求系统信息、网络诊断等能力。

This isn’t Alexa.
This isn’t ChatGPT.
This is **SoulSpeak**.
