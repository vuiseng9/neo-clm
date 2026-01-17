import typer
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch

# NOTE: disable pretty exceptions to make errors native like bare python stacktrace
app = typer.Typer(
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False)

def divstr(len=50, nl=True):
    return "─" * len + ("\n" if nl else "")

@app.command()
def main(
    model_id_or_path: str = typer.Argument(..., help="Path to checkpoint directory or HuggingFace model ID"),
    max_new_tokens: int = typer.Option(64, "-m", help="Maximum new tokens to generate"),
    prompt_ids: str = typer.Option("0,4,8", "-p", help="Comma-separated indices (e.g., '0,4,8') or 'all' to use first n samples"),
    trust_remote_code: bool = typer.Option(False, "-R", help="Whether to trust remote code when loading model")
):

    # Load HuggingFace model
    model = AutoModelForCausalLM.from_pretrained(model_id_or_path, device_map="auto", trust_remote_code=trust_remote_code)
    tokzr = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
    tokzr.padding_side = "left" # causal lm padding side

    # Load test prompts
    script_dir = Path(__file__).parent
    testdf = pd.read_json(script_dir / "prompts.json")
    
    # Select prompts based on prompt_ids
    if prompt_ids.lower() == "all":
        prompts = testdf['prompt'].values.tolist()
    else:
        idx_list = [int(i.strip()) for i in prompt_ids.split(",")]
        prompts = testdf['prompt'].values[idx_list].tolist()

    ngen = len(prompts)
    enc = tokzr(prompts)
    
    # NOTE: intentionally generate sequences one by one to 
    # avoid handling complexities with varying lengths and padding in batch generation

    for i, input_ids in enumerate(enc['input_ids']):
        inputs = dict(
            input_ids = torch.tensor([input_ids], device=model.device),
            attention_mask = torch.tensor([enc['attention_mask'][i]], device=model.device),
        )
        output =model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1)
        decoded = tokzr.decode(output[0], skip_special_tokens=False)
        print(f"\n── Prompt {i+1}/{ngen} | total tokens = {len(output[0])} " + divstr())
        pos = len(prompts[i])
        print(f"{decoded[:pos]} {divstr(5)}{decoded[pos:]}")
        print(decoded)

    print("\n\n── end " + "─" * 50)

if __name__ == "__main__":
    app()