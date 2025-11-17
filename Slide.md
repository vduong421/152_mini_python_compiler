# Mini-Python Compiler — Full Slide Deck (Markdown)

## Slide 1 — Title Slide
### Mini-Python Subset Compiler
- A small compiler for a restricted subset of Python  
- Supports assignments, arithmetic, if/else, while-loops, and print  
- Produces custom stack-machine assembly  
- Executes via our virtual machine  
- Created for CMPE 152 Final Project  

## Slide 2 — Agenda
- Regular Languages & DFA  
- Lexical Analysis (Tokenization)  
- CFG (Context-Free Grammar)  
- CNF & GNF conversion  
- Parse trees & derivations  
- AST construction  
- Three working test programs  
- Code generation  
- Virtual machine & execution  
- Error handling  
- Limitations  

## Slide 3 — Regular Expressions for Tokens
We define the lexical structure using regular expressions.

```python
TOKEN_SPEC = [
    ("NUMBER",   r"\d+"),
    ("ID",       r"[A-Za-z_][A-Za-z0-9_]*"),
    ("EQ",       r"=="),
    ("NE",       r"!="),
    ("LE",       r"<="),
    ("GE",       r">="),
    ("LT",       r"<"),
    ("GT",       r">"),
    ("PLUS",     r"\+"),
    ("MINUS",    r"-"),
    ("MUL",      r"\*"),
    ("DIV",      r"/"),
    ("ASSIGN",   r"="),
    ("COLON",    r":"),
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
]
```

## Slide 4 — DFA for Identifiers vs Numbers
```text
States: q0 (start), q_id, q_num, q_err

q0:
  letter/_ → q_id
  digit → q_num
  other → q_err

q_id:
  letter/digit/_ → q_id
  other → accept ID

q_num:
  digit → q_num
  other → accept NUMBER
```

## Slide 5 — Tokenization Algorithm
```python
MASTER_RE = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPEC)
)
```

## Slide 6 — Example Tokenization Output
```text
(ID:x) (ASSIGN:=) (NUMBER:1) (PLUS:+) (NUMBER:2)
(MUL:*) (NUMBER:3) (NEWLINE)
```

## Slide 7 — CFG
```text
Program  -> StmtList
StmtList -> Stmt StmtList | ε
Stmt     -> SimpleStmt NEWLINE | IfStmt | WhileStmt
SimpleStmt -> AssignStmt | PrintStmt
AssignStmt -> ID ASSIGN Expr
```

## Slide 8 — CNF Sample
```text
S -> IF_COND
IF_COND -> IF_EXPR COLON_BLOCK
COLON_BLOCK -> COLON BLOCK
```

## Slide 9 — GNF Sample
```text
Expr -> NUMBER ExprTail
ExprTail -> PLUS Expr | MINUS Expr | ε
```

## Slide 10 — Parse Tree (Demo 1)
```text
Program
  Assign(x)
    BinOp(PLUS)
      Number(1)
      BinOp(MUL)
        Number(2)
        Number(3)
```

## Slide 11 — Leftmost Derivation Example
```text
Program ⇒ StmtList
⇒ Stmt StmtList
⇒ AssignStmt NEWLINE StmtList
```

## Slide 12 — AST Construction Code
```python
class BinOp(AST):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right
```

## Slide 13 — Demo 1 Source
```python
x = 1 + 2 * 3
y = x + 4
print(y)
```

## Slide 14 — Demo 1 Tokens
```text
(ID:x) (ASSIGN:=) (NUMBER:1) (PLUS:+)
```

## Slide 15 — Demo 1 AST
```text
Assign(x)
  BinOp(PLUS)
    Number(1)
    BinOp(MUL)
```

## Slide 16 — Demo 1 Assembly
```text
000: PUSH_CONST 1
001: PUSH_CONST 2
002: PUSH_CONST 3
003: MUL
004: ADD
005: STORE_VAR x
```

## Slide 17 — Demo 1 Execution
```text
[VM PRINT] 11
env = {'x': 7, 'y': 11}
```

## Slide 18 — Demo 2 Source
```python
x = 0
while x < 5:
    x = x + 1
print(x)
```

## Slide 19 — Demo 2 Assembly
```text
002: LABEL LOOP0
003: LOAD_VAR x
004: PUSH_CONST 5
005: CMP_LT
006: JUMP_IF_FALSE END1
```

## Slide 20 — Demo 2 Execution
```text
[VM PRINT] 5
Final env: {'x': 5}
```

## Slide 21 — Demo 3 Source
```python
x = 10
if x > 5:
    print(x)
else:
    print(0)
```

## Slide 22 — Demo 3 Assembly
```text
005: JUMP_IF_FALSE ELSE0
006: LOAD_VAR x
007: PRINT
```

## Slide 23 — Demo 3 Execution
```text
[VM PRINT] 10
```

## Slide 24 — Virtual Machine Design
```python
stack = []
env = {}
pc = 0
```

## Slide 25 — VM Instruction Set
```text
PUSH_CONST n
LOAD_VAR x
STORE_VAR x
ADD, MUL
CMP_LT, CMP_GT
JUMP, JUMP_IF_FALSE
PRINT
```

## Slide 26 — VM Run Loop
```python
while pc < len(code):
    op = code[pc]
    pc = execute(op, pc)
```

## Slide 27 — Dataflow Overview
- Expressions flow through stack  
- STORE_VAR updates environment  
- Control-flow uses jump labels  

## Slide 28 — Memory Model (Environment)
```text
env = {"x": 10, "y": 4}
```

## Slide 29 — Error Handling Example
```text
Parser errors:
- Unexpected token NEWLINE at line 1
- Expected COLON, got NEWLINE at line 2
```

## Slide 30 — Error Handling Code
```python
def error(msg, line):
    errors.append(f"{msg} at line {line}")
```

## Slide 31 — Lexical Errors
Handled by regex mismatch.

## Slide 32 — Parsing Errors
- Missing colon  
- Unexpected token  
- Wrong expression  

## Slide 33 — Code Generation Errors
- AST node missing  
- Invalid operator  

## Slide 34 — Summary of Contributions
- Fully working compiler  
- Three demos  
- Virtual machine  
- AST + CFG + CNF + GNF  
- Error handling  

## Slide 35 — Limitations
- No functions  
- No real Python features  
- No type system  
- No optimizations  

## Slide 36 — Future Work
- Add functions  
- Add types  
- Add optimizer  
- Compile to real assembly  

## Slide 37 — Conclusion
We successfully built a full compiler pipeline.

## Slide 38 — Questions?
Thank you!
