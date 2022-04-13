from typing import List, Any

import torch
from torch import nn


# noinspection PyMethodMayBeStatic,PyShadowingBuiltins,PyTypeHints
class CharacterTokenizer(nn.Module):
    def forward(self, input: Any) -> Any:
        """
        :param input: Input sentence or list of sentences on which to apply tokenizer.
        :type input: Union[str, List[str]]
        :return: tokenized text
        :rtype: Union[List[str], List[List(str)]]
        """
        if torch.jit.isinstance(input, List[str]):
            tokens: List[List[str]] = []
            for text in input:
                tokens.append(list(text))
            return tokens
        elif torch.jit.isinstance(input, str):
            return list(input)
        else:
            raise TypeError("Input type not supported")
