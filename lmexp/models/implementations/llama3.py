from transformers import AutoTokenizer, AutoModelForCausalLM
from lmexp.generic.hooked_model import HookedModel
from lmexp.generic.tokenizer import Tokenizer
import torch


class LlamaTokenizer(Tokenizer):
    def __init__(self, model_name=MODEL_LLAMA_3):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors="pt")

    def decode(self, tensor):
        return self.tokenizer.decode(tensor, skip_special_tokens=True)

class ProbedLlama3(HookedModel):
    def __init__(self, model_name=MODEL_LLAMA_3):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.end_of_instruction = MODEL_ID_TO_END_OF_INSTRUCTION.get(model_name, "")

    def get_n_layers(self):
        return len(self.model.model.layers)

    def forward(self, x: torch.tensor):
        return self.model(x)

    def sample(self, tokens: torch.tensor, max_n_tokens: int) -> torch.tensor:
        attention_mask = torch.ones_like(tokens)
        return self.model.generate(
            tokens,
            max_length=max_n_tokens,
            attention_mask=attention_mask,
            pad_token_id=self.model.config.eos_token_id,
        )

    def resid_dim(self) -> int:
        return self.model.config.hidden_size

    def get_module_for_layer(self, layer: int) -> torch.nn.Module:
        return self.model.model.layers[layer]

    def prepare_input(self, input_text: str) -> str:
        if self.end_of_instruction:
            return f"{input_text}{self.end_of_instruction}"
        return input_text