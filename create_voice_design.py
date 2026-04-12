import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device.startswith("cuda") else torch.float16

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map=device,
    dtype=dtype,
    low_cpu_mem_usage=True,
)

# single inference
wavs, sr = model.generate_voice_design(
    text="""
    We all think with this (brain image)

    But most people rely thinking through instinct rather than mindset:

    He’s wrong, that’s not fair, whatever, these are instinct.

    A mindset tells you where the problem is.

    For example, when someone said, “I wanna be free”,

    then the question might be,

    “What is freedom?”

    “What does it mean to be free?”

    “Why do we want to be free?”

    “Where does freedom come from?”

    And you say, “well, what’s the point in thinking about these?”

    And the answer is that, “there isn’t, at least in the short term”

    But in the long run, a mindset is the process of formulating the understanding of ourselves and the world.

    And the study of this process is Philosophy.

    Philosophy consists of:

    1. Metaphysics
    2. Epistemology
    3. Ethics
    4. Logic
    5. Philosophy of Mind
    6. Philosophy of Language
    7. Philosophy of Religion
    8. Political Philosophy
    9. Aesthetics

    Why does it matter?

    Because to be able to contribute sensibly to the society, having such a mindset is essential.

    Why do we need to contribute?

    Because it is only through contribution, no matter how small, do we cultivate meanings in life.

    And meanings are essential to living a sensible life.
    """,
    language="English",
    instruct="""
    A deep, weathered male voice in his late 60s to early 70s. The tone carries the texture of lived experience — slightly rough around the edges, yet steady and controlled. Not frail, but deliberate. Every sentence feels considered before it is spoken.

    The pacing is unhurried, with natural pauses that allow ideas to settle. Words are articulated clearly, but without theatrical exaggeration. There is a quiet gravity to the delivery — the kind that suggests reflection rather than performance.

    The emotional register is calm and contemplative, with subtle warmth beneath a reserved exterior. When discussing difficult truths, the voice does not intensify; instead, it lowers slightly, becoming more intimate, as if sharing something confidential with the listener.

    No dramatic crescendos. No motivational hype. The authority comes from restraint.

    Accent: Neutral American English (slightly textured, non-regional).
    Cadence: Measured, rhythmic, almost meditative.
    Energy: Low to medium.
    Impression: A retired professor, a monk who once studied philosophy, or a grandfather who has spent decades observing human nature.
    """,
)
sf.write("what_is_philosohpy_0410.wav", wavs[0], sr)

# # batch inference
# wavs, sr = model.generate_voice_design(
#     text=[
#         "We all think with this",
#         "But most people rely thinking through instinct rather than mindset",
#         "He's wrong, that's stupid, whatever, these are instinct",
#         "A mindset tells you where the problem is.",
#         "For example, when someone said, I wanna be free",
#         "then the question might be",
#         "What is freedom?",
#         "What does it mean to be free?",
#         "Why do we want to be free?",
#         "Where does freedom come from?",
#         "And you say, well, what's the point in thinking about these?",
#         "And the answer is that, theere isn't, at least in the short term",
#         "But in the long run, there could be, for gaining a mindset.",
#         "And a mindset is the process of formulating the understanding of ourselves and the world.",
#         "And the study of this process is called Philosophy.",
#         "Philosophy consists of metaphysics, epistemology, ethics, logics, philosophy of mind, philosophy of language, philosophy of religion, political philosophy, and aesthetics.",
#         "Why do they matter?",
#         "Because to be able to contribute sensibly to the society, having such a mindset is essential.",
#         "Why do we need to contribute?",
#         "Because it is only through contribution, no matter how small, do we cultivate meanings in life.",
#         "And meanings are essential to living a sensible life."
#     ],
#     language=["English"],
#     instruct=[
#         """
#         A deep, weathered male voice in his late 60s to early 70s. The tone carries the texture of lived experience — slightly rough around the edges, yet steady and controlled. Not frail, but deliberate. Every sentence feels considered before it is spoken.
#         The pacing is unhurried, with natural pauses that allow ideas to settle. Words are articulated clearly, but without theatrical exaggeration. There is a quiet gravity to the delivery — the kind that suggests reflection rather than performance.
#         The emotional register is calm and contemplative, with subtle warmth beneath a reserved exterior. When discussing difficult truths, the voice does not intensify; instead, it lowers slightly, becoming more intimate, as if sharing something confidential with the listener.
#         No dramatic crescendos. No motivational hype. The authority comes from restraint.
#         Accent: Neutral American English (slightly textured, non-regional).
#         Cadence: Measured, rhythmic, almost meditative.
#         Energy: Low to medium.
#         Impression: A retired professor, a monk who once studied philosophy, or a grandfather who has spent decades observing human nature.
#         """
#     ],
# )
# sf.write("what_is_philosophy_1.wav", wavs[0], sr)
# sf.write("what_is_philosophy_2.wav", wavs[1], sr)
# sf.write("what_is_philosophy_3.wav", wavs[2], sr)
# sf.write("what_is_philosophy_4.wav", wavs[3], sr)
# sf.write("what_is_philosophy_5.wav", wavs[4], sr)
# sf.write("what_is_philosophy_6.wav", wavs[5], sr)
# sf.write("what_is_philosophy_7.wav", wavs[6], sr)
# sf.write("what_is_philosophy_8.wav", wavs[7], sr)
# sf.write("what_is_philosophy_9.wav", wavs[8], sr)
# sf.write("what_is_philosophy_10.wav", wavs[9], sr)
# sf.write("what_is_philosophy_11.wav", wavs[10], sr)
# sf.write("what_is_philosophy_12.wav", wavs[11], sr)
# sf.write("what_is_philosophy_13.wav", wavs[12], sr)
# sf.write("what_is_philosophy_14.wav", wavs[13], sr)
# sf.write("what_is_philosophy_15.wav", wavs[14], sr)
# sf.write("what_is_philosophy_16.wav", wavs[15], sr)
# sf.write("what_is_philosophy_17.wav", wavs[16], sr)
# sf.write("what_is_philosophy_18.wav", wavs[17], sr)
# sf.write("what_is_philosophy_19.wav", wavs[18], sr)
# sf.write("what_is_philosophy_20.wav", wavs[19], sr)
# sf.write("what_is_philosophy_21.wav", wavs[20], sr)
# sf.write("what_is_philosophy_22.wav", wavs[21], sr)
