from lm_eval import simple_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from pathlib import Path
import json, argparse

from examples.llama3 import build_transformer, Tokenizer, prefill, MODEL_PARAMS
from tinygrad.tensor import Tensor
from tinygrad import Device

class LLaMaAdaptor(LM):
  def __init__(
    self,
    model_size: str,
    checkpoint_path: Path,
    is_chat_model: bool,
    max_length: int = 128000,
    quantize: bool = False,
  ):
    super().__init__()
    self.max_length = max_length
    self.is_chat_model = is_chat_model
    self.tokenizer = Tokenizer(str((checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent) / "tokenizer.model"))
    self.model = build_transformer(checkpoint_path, model_size, quantize)
  def generate_until(self, requests: list[Instance]) -> list[str]:
    continuations = []
    for request in requests:
      prompt, args = request.args
      temperature = args.get("temperature", 0.0)
      max_length = args.get("max_length", self.max_length)
      until = [self.tokenizer.encode(tok) for tok in args.get("until", [])]
      if self.is_chat_model:
        toks = [self.tokenizer.bos_id] + self.tokenizer.encode_message("system", "You are a helpful assistant") + \
          self.tokenizer.encode_message("user", prompt) + self.tokenizer.encode_role("assistant")
      else:
        toks = [self.tokenizer.bos_id] + self.tokenizer.encode(prompt)
      prefix_length = len(toks)
      prefill(self.model, toks[:-1])
      start_pos = prefix_length-1
      for i in range(max_length):
        next_tok = self.model(Tensor([toks[start_pos:]]), start_pos, temperature, top_k=1).item()
        start_pos = len(toks)
        toks.append(next_tok)
        if next_tok in self.tokenizer.stop_tokens or next_tok in until: break
      continuations.append(self.tokenizer.decode(toks[prefix_length:]))
    return continuations
  def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]: raise NotImplementedError()
  def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]: raise NotImplementedError()

if __name__ == '__main__':
  print(f"using {Device.DEFAULT} backend")

  parser = argparse.ArgumentParser(description='Run LLaMA evals in tinygrad', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--size', type=str, default="8B", help=f"Size of model to use [{', '.join(list(MODEL_PARAMS.keys()))}]")
  parser.add_argument('--chat', action='store_true', help="Use chat formatting")
  parser.add_argument('--quantize', type=str, default=None, help="Quantize the weights to int8 or int4 in memory")
  parser.add_argument('--task', type=str, default="gsm8k_cot_llama", help="lm_eval task")
  parser.add_argument('--limit', type=int, default=None, help="Limit tests in eval")
  parser.add_argument('--num-fewshot', type=int, default=None, help="Limit tries(starts with 0)")
  parser.add_argument('--weights', type=str, default="./weights/LLaMa/", help="Location of the weights")
  parser.add_argument('--output-path', type=str, default="./result.json", help="Location of file with results")
  args = parser.parse_args()

  # run eval and exit
  adaptor = LLaMaAdaptor(model_size=args.size, quantize=args.quantize,
                         checkpoint_path=Path(args.weights), is_chat_model=args.chat)
  results = simple_evaluate(
    model=adaptor,
    tasks=args.task.split(","),
    num_fewshot=args.num_fewshot,
    task_manager=None,
    limit=args.limit
  )
  Path(args.output_path).write_text(json.dumps(results, indent=2))
