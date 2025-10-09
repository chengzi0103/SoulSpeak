"""Keyboard-driven dialogue using OpenAI-native agent with streamed TTS playback."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from soul_speak.llm.openai_native import build_agent
from soul_speak.llm.llm_tagged import _split_for_tts
from soul_speak.llm.run_and_speak import play_sentences

logger = logging.getLogger(__name__)


async def interactive_loop() -> None:
    load_dotenv()
    agent = build_agent()
    print("Emilia (原生代理 + TTS) — 输入 exit 退出")

    while True:
        try:
            user_input = input("你: ").strip()
        except EOFError:
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            break

        reply = await agent.generate(user_input)
        print(f"Emilia: {reply}")

        sentences: List[str] = _split_for_tts(reply)
        await play_sentences(sentences)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    asyncio.run(interactive_loop())


if __name__ == "__main__":
    main()
