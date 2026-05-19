#pragma once

#include <string>

namespace devproc2 {

// Register tokenizer packed funcs. When tokenizers-cpp support is enabled,
// this registers runtime.tokenizer.paligemma_encode.
void RegisterTokenizerPackedFuncs();

// Configure the default PaliGemma SentencePiece model used by the packed func.
void SetPaligemmaTokenizerModelPath(std::string path);

}  // namespace devproc2
