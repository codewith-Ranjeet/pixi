import torch
from tokenizers import Tokenizer
from src.model import Pixi, PixiConfig
from src.export_and_infer import generate_chat


CHECKPOINT = "checkpoints/pretrain/pixi_step_10000.pt"
TOKENIZER_PATH = "tokenizer/tokenizer.json"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Pixi-40M on {device}...")

    # Load the checkpoint saved after pretraining.
    # This assumes the checkpoint was created by your own trusted code.
    torch.serialization.add_safe_globals([PixiConfig])
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    # Reconstruct the model using the saved configuration.
    saved_config = checkpoint["config"]

    if isinstance(saved_config, dict):
        config = PixiConfig(**saved_config)
    elif isinstance(saved_config, PixiConfig):
        config = saved_config
    else:
        raise TypeError(
            f"Unexpected checkpoint config type: {type(saved_config).__name__}"
        )

    model = Pixi(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    print(f"Checkpoint step: {checkpoint['step']}")
    print(f"Saved training loss: {checkpoint['loss']}")
    print("Pixi is ready! Type /exit to quit.\n")

    while True:
        user_text = input("You: ").strip()

        if user_text.lower() in {"/exit", "/quit"}:
            print("Goodbye!")
            break

        if not user_text:
            continue

        # Format the input using Pixi's ChatML-style conversation format.
        prompt = (
            "<|im_start|>user\n"
            f"{user_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        prompt_ids = tokenizer.encode(prompt).ids

        # Avoid exceeding the model's context window.
        max_prompt_tokens = config.max_seq_len - 256
        if len(prompt_ids) > max_prompt_tokens:
            prompt_ids = prompt_ids[-max_prompt_tokens:]

        with torch.inference_mode():
            output_ids = generate_chat(
                model,
                prompt_ids,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                device=device,
            )

        # Hide special tokens in the displayed response.
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        print(f"Pixi: {response}\n")


if __name__ == "__main__":
    main()