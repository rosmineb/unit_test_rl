import argparse
import xgrammar as xgr
from transformers import AutoTokenizer, AutoConfig
import pdb

def parse_arguments():
    """
    Parse command line arguments for grammar checking.
    
    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Check if a string conforms to a specified grammar.")
    parser.add_argument("--grammar_file", type=str, help="Path to the file containing the grammar definition.")
    parser.add_argument("--string_file", type=str, help="Path to the file containing the string to check against the grammar.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model ID to use for tokenization.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    with open(args.string_file, 'r') as f:
        candidate = f.read().strip()
        print(f"candidate: {candidate}")

    with open(args.grammar_file, 'r') as f:
        ebnf_source = f.read()

    print(f'Grammar: \n{ebnf_source}')

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    full_vocab_size = AutoConfig.from_pretrained(args.model_id).vocab_size
    tok_info = xgr.TokenizerInfo.from_huggingface(tokenizer,
                                                  vocab_size=full_vocab_size)

    compiler = xgr.GrammarCompiler(tok_info) #, allow_isolated_special=True)
    # three common options ⬇︎
    # compiled = compiler.compile_builtin_json_grammar()            # built-in JSON
    # compiled = compiler.compile_json_schema(schema_str)         # JSON schema
    compiled = compiler.compile_grammar(ebnf_source)            # any EBNF text

    matcher = xgr.GrammarMatcher(compiled)

    token_ids  = tokenizer.encode(candidate, add_special_tokens=False)

    ok = True
    good_up_to = []
    failed_token = None
    for tok in token_ids:
        if not matcher.accept_token(tok):          # returns False = mismatch
            ok = False
            failed_token = tok
            break
        else:
            good_up_to.append(tok)

    # the string matches the grammar **iff**
    #   • every token was accepted, and
    #   • the matcher reached a terminal state.
    ok = ok and matcher.is_terminated()            # end must be legal
    print("valid!" if ok else "invalid!")
    if not ok:
        print("parsed up to:")
        print(tokenizer.decode(good_up_to))
        print(f"failed token: {tokenizer.decode(failed_token)} {failed_token}")
        pdb.set_trace()

    print(f"ok: {ok}")