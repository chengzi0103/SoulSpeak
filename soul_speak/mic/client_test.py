# test_audio_playback.py

import sounddevice as sd
import numpy as np

# 音频参数
samplerate = 44100  # 可以使用一个标准采样率进行测试
duration = 3.0      # 播放时长 (秒)
frequency = 440.0   # 频率 (Hz)，这是一个标准的A4音

print("--- 正在测试本地音频播放 ---")

try:
    # 生成一个简单的正弦波，模拟音频数据
    t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
    amplitude = 0.5 # 振幅，范围 -1.0 到 1.0
    audio_data = amplitude * np.sin(2. * np.pi * frequency * t)

    # 播放音频。如果你的耳机连接正常，应该能听到一个持续3秒的“嘟”声。
    print(f"正在以 {samplerate} Hz 播放 {duration} 秒的 {frequency} Hz 测试音...")
    sd.play(audio_data, samplerate)
    sd.wait() # 等待播放完成
    print("--- 音频播放测试完成 ---")
    print("如果听到了声音，说明 sounddevice 和音频设备工作正常。")
    print("如果未听到声音或程序报错，请检查耳机连接、系统音频设置以及 PortAudio 库的安装。")

except Exception as e:
    print(f"--- 音频播放测试失败：{e} ---")
    print("请检查：")
    print("1. 确保你的耳机已连接并设置为默认音频输出设备。")
    print("2. 确保系统中安装了 PortAudio (sounddevice 的底层库)。")
    print("   - Linux (Ubuntu/Debian): sudo apt-get install libportaudio2")
    print("   - Windows/macOS: sounddevice 通常会自动安装，如果不行，尝试重新安装 sounddevice。")
    print("3. 尝试重启电脑。")