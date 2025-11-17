# Mini-Python Compiler – Description & Code Mapping

This document connects each **presentation title/requirement** to the **exact code snippet** in `main.py` that supports it.  
You can use this as notes when you build your slides.

---

## 1. Lexical Analysis, Regular Expressions, Token Classes

**What this proves**

- You are using **regular expressions** to define a **regular language** for your tokens.
- You classify tokens into **keywords, identifiers, numbers, operators, punctuation, whitespace, comments**, etc.

**Key code**

```python
KEYWORDS = {"if", "else", "while", "print"}

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
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
    ("COLON",    r":"),
    ("SKIP",     r"[ \t]+"),
    ("COMMENT",  r"\#.*"),
    ("MISMATCH", r"."),
]

MASTER_RE = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPEC))
```

```python
@dataclass
class Token:
    type: str
    value: Any
    line: int
    column: int
```

```python
def lex(source: str) -> List[Token]:
    tokens: List[Token] = []
    indent_stack = [0]

    lines = source.splitlines()
    for lineno, raw_line in enumerate(lines, start=1):
        # indentation handling omitted here for brevity ...

        # Tokenize the rest of the line
        pos = len(raw_line) - len(stripped)
        while pos < len(raw_line):
            mo = MASTER_RE.match(raw_line, pos)
            if not mo:
                raise LexerError(f"Illegal character at line {lineno}: {raw_line[pos]!r}")
            kind = mo.lastgroup
            value = mo.group()
            col = mo.start() + 1
            if kind == "NUMBER":
                tokens.append(Token("NUMBER", int(value), lineno, col))
            elif kind == "ID":
                if value in KEYWORDS:
                    tokens.append(Token(value.upper(), value, lineno, col))
                else:
                    tokens.append(Token("ID", value, lineno, col))
            elif kind in {"EQ", "NE", "LE", "GE", "LT", "GT",
                          "PLUS", "MINUS", "MUL", "DIV",
                          "ASSIGN", "LPAREN", "RPAREN", "COLON"}:
                tokens.append(Token(kind, value, lineno, col))
            elif kind == "SKIP":
                pass
            elif kind == "COMMENT":
                break
            elif kind == "MISMATCH":
                raise LexerError(f"Unexpected character {value!r} at line {lineno}, col {col}")
            pos = mo.end()
        tokens.append(Token("NEWLINE", "\n", lineno, len(raw_line) + 1))

    # close indent levels ...
    tokens.append(Token("EOF", "", lineno, 0))
    return tokens
```

---

## 2. Context-Free Grammar (CFG) and Context-Free Language

**What this proves**

- You explicitly define a **Context-Free Grammar (CFG)** for your mini-Python language.
- You can use this on a slide to show the **CFL** you are recognizing.

**Key code**

```python
GRAMMAR_TEXT = r"""
Program     -> StmtList
StmtList    -> Stmt StmtList | ε
Stmt        -> SimpleStmt NEWLINE | IfStmt | WhileStmt
SimpleStmt  -> AssignStmt | PrintStmt
AssignStmt  -> ID ASSIGN Expr
PrintStmt   -> PRINT LPAREN Expr RPAREN
IfStmt      -> IF Expr COLON NEWLINE INDENT StmtList DEDENT
             | IF Expr COLON NEWLINE INDENT StmtList DEDENT ELSE COLON NEWLINE INDENT StmtList DEDENT
WhileStmt   -> WHILE Expr COLON NEWLINE INDENT StmtList DEDENT
Expr        -> Expr EQ Expr
             | Expr NE Expr
             | Expr LT Expr
             | Expr LE Expr
             | Expr GT Expr
             | Expr GE Expr
             | Expr PLUS Term
             | Expr MINUS Term
             | Term
Term        -> Term MUL Factor
             | Term DIV Factor
             | Factor
Factor      -> PLUS Factor
             | MINUS Factor
             | Primary
Primary     -> NUMBER
             | ID
             | LPAREN Expr RPAREN
"""
```

You also **implement** this grammar in code using recursive descent:

```python
class Parser:
    def parse(self) -> Program:
        body = []
        while self.current().type != "EOF":
            if self.current().type == "NEWLINE":
                self.advance()
                continue
            stmt = self.parse_statement()
            if stmt is not None:
                body.append(stmt)
        return Program(body)

    def parse_statement(self) -> Optional[AST]:
        tok = self.current()
        if tok.type == "IF":
            return self.parse_if()
        elif tok.type == "WHILE":
            return self.parse_while()
        elif tok.type == "PRINT":
            return self.parse_print()
        elif tok.type == "ID":
            if self.tokens[self.pos + 1].type == "ASSIGN":
                return self.parse_assign()
            # error handling omitted here ...
```

---

## 3. CNF and GNF (Chomsky & Greibach Normal Forms)

**What this proves**

- You can convert **parts of your grammar** into **CNF** and **GNF** forms.
- Even if not fully parsed by code, this **meets the theory requirement**.

**Key code**

```python
CNF_SNIPPET = r"""
Example CNF for a tiny subset:

S  -> IF_COND
IF_COND -> IF_EXPR COLON_BLOCK
IF_EXPR -> IF Expr
COLON_BLOCK -> COLON BLOCK
BLOCK -> NEWLINE INDENT StmtList DEDENT

(All productions have at most 2 non-terminals or 1 terminal.)
"""
```

```python
GNF_SNIPPET = r"""
Example GNF for expression grammar subset:

Expr -> NUMBER ExprTail
      | ID ExprTail
ExprTail -> PLUS Expr
          | MINUS Expr
          | EQ Expr
          | ε

(Each production starts with a terminal, followed by non-terminals.)
"""
```

These strings are printed in `show_theory()` for your slides.

---

## 4. DFA / Regular Language for Lexical Syntax

**What this proves**

- You understand how your **regular expressions** correspond to a **DFA**.
- You can draw this DFA on a slide using this description.

**Key code**

```python
DFA_DESCRIPTION = r"""
Example DFA (for identifiers vs numbers):

States: q0 (start), q_id, q_num, q_err
Alphabet: letters, digits, underscore

From q0:
  letter or '_' -> q_id
  digit         -> q_num
  other         -> q_err

From q_id:
  letter/digit/'_' -> q_id
  other            -> accept ID (and return to q0 for next token)

From q_num:
  digit -> q_num
  other -> accept NUMBER (and return to q0)

q_err is a trap state.
"""
```

Printed by `show_theory()`.

---

## 5. Tokenization + Parsing + AST (Internal Operation)

**What this proves**

- You build a real **AST** and show internal compiler structure.
- You can screenshot AST output for your slides.

**AST node definitions**

```python
class AST:
    pass

@dataclass
class Program(AST):
    body: List[AST]

@dataclass
class Number(AST):
    value: int
    line: int

@dataclass
class Var(AST):
    name: str
    line: int

@dataclass
class BinOp(AST):
    op: str
    left: AST
    right: AST
    line: int

@dataclass
class UnaryOp(AST):
    op: str
    operand: AST
    line: int

@dataclass
class Assign(AST):
    name: str
    expr: AST
    line: int

@dataclass
class Print(AST):
    expr: AST
    line: int

@dataclass
class If(AST):
    cond: AST
    then_body: List[AST]
    else_body: Optional[List[AST]]
    line: int

@dataclass
class While(AST):
    cond: AST
    body: List[AST]
    line: int
```

**AST pretty-printer**

```python
def format_ast(node: AST, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(node, Program):
        lines = [pad + "Program"]
        for stmt in node.body:
            lines.append(format_ast(stmt, indent + 1))
        return "\n".join(lines)
    if isinstance(node, Number):
        return pad + f"Number({node.value})"
    if isinstance(node, Var):
        return pad + f"Var({node.name})"
    if isinstance(node, BinOp):
        lines = [pad + f"BinOp({node.op})"]
        lines.append(format_ast(node.left, indent + 1))
        lines.append(format_ast(node.right, indent + 1))
        return "\n".join(lines)
    # ... Print, If, While, etc.
```

---

## 6. If / While Parsing (Control Structures)

**What this proves**

- You have grammar rules and parser logic for **control flow**: `if`, `else`, `while`.
- You can show parse trees for these constructs.

**Key code**

```python
def parse_if(self) -> If:
    tok_if = self.expect("IF")
    cond = self.parse_expression()
    self.expect("COLON")
    self.expect("NEWLINE")
    then_body = self.parse_block()
    else_body = None
    if self.current().type == "ELSE":
        self.advance()
        self.expect("COLON")
        self.expect("NEWLINE")
        else_body = self.parse_block()
    return If(cond, then_body, else_body, tok_if.line)
```

```python
def parse_while(self) -> While:
    tok_w = self.expect("WHILE")
    cond = self.parse_expression()
    self.expect("COLON")
    self.expect("NEWLINE")
    body = self.parse_block()
    return While(cond, body, tok_w.line)
```

```python
def parse_block(self) -> List[AST]:
    self.expect("INDENT")
    stmts: List[AST] = []
    while self.current().type not in {"DEDENT", "EOF"}:
        if self.current().type == "NEWLINE":
            self.advance()
            continue
        stmt = self.parse_statement()
        if stmt is not None:
            stmts.append(stmt)
    self.expect("DEDENT")
    return stmts
```

---

## 7. Code Generation: Custom Assembly Output

**What this proves**

- You convert AST into a **low-level target language** (your custom stack assembly).
- This is your **code generation** phase.

**Key code**

```python
@dataclass
class Instr:
    op: str
    args: Tuple[Any, ...] = ()
```

```python
def generate_program(prog: Program) -> List[Instr]:
    code: List[Instr] = []
    label_counter = 0

    def new_label(prefix: str = "L") -> str:
        nonlocal label_counter
        label = f"{prefix}{label_counter}"
        label_counter += 1
        return label

    def gen_expr(node: AST):
        if isinstance(node, Number):
            code.append(Instr("PUSH_CONST", (node.value,)))
        elif isinstance(node, Var):
            code.append(Instr("LOAD_VAR", (node.name,)))
        elif isinstance(node, UnaryOp):
            gen_expr(node.operand)
            if node.op == "MINUS":
                code.append(Instr("NEG", ()))
        elif isinstance(node, BinOp):
            gen_expr(node.left)
            gen_expr(node.right)
            op_map = {
                "PLUS": "ADD",
                "MINUS": "SUB",
                "MUL": "MUL",
                "DIV": "DIV",
                "LT": "CMP_LT",
                "LE": "CMP_LE",
                "GT": "CMP_GT",
                "GE": "CMP_GE",
                "EQ": "CMP_EQ",
                "NE": "CMP_NE",
            }
            asm_op = op_map[node.op]
            code.append(Instr(asm_op, ()))
```

```python
    def gen_stmt(node: AST):
        if isinstance(node, Assign):
            gen_expr(node.expr)
            code.append(Instr("STORE_VAR", (node.name,)))
        elif isinstance(node, Print):
            gen_expr(node.expr)
            code.append(Instr("PRINT", ()))
        elif isinstance(node, If):
            else_label = new_label("ELSE")
            end_label = new_label("END")
            gen_expr(node.cond)
            code.append(Instr("JUMP_IF_FALSE", (else_label,)))
            for s in node.then_body:
                gen_stmt(s)
            code.append(Instr("JUMP", (end_label,)))
            code.append(Instr("LABEL", (else_label,)))
            if node.else_body:
                for s in node.else_body:
                    gen_stmt(s)
            code.append(Instr("LABEL", (end_label,)))
        elif isinstance(node, While):
            start_label = new_label("LOOP")
            end_label = new_label("END")
            code.append(Instr("LABEL", (start_label,)))
            gen_expr(node.cond)
            code.append(Instr("JUMP_IF_FALSE", (end_label,)))
            for s in node.body:
                gen_stmt(s)
            code.append(Instr("JUMP", (start_label,)))
            code.append(Instr("LABEL", (end_label,)))
```

```python
    for stmt in prog.body:
        gen_stmt(stmt)
    return code
```

---

## 8. Result of Compilation: Execution & Final Values (3 Test Cases)

**What this proves**

- Your compiler not only generates code but also **runs** it with a small VM.
- You show the **result of compilation** for **three test programs**.

**Virtual Machine / execution engine**

```python
def run_program(code: List[Instr]) -> Dict[str, Any]:
    label_to_pc: Dict[str, int] = {}
    for idx, instr in enumerate(code):
        if instr.op == "LABEL":
            label_name = instr.args[0]
            label_to_pc[label_name] = idx

    patched_code: List[Instr] = []
    for instr in code:
        if instr.op in {"JUMP", "JUMP_IF_FALSE"}:
            label = instr.args[0]
            target_pc = label_to_pc[label]
            patched_code.append(Instr(instr.op, (target_pc,)))
        else:
            patched_code.append(instr)

    code = patched_code

    stack: List[Any] = []
    env: Dict[str, Any] = {}
    pc = 0

    def pop():
        if not stack:
            raise RuntimeError("Stack underflow")
        return stack.pop()
```

```python
    while pc < len(code):
        instr = code[pc]
        op = instr.op
        args = instr.args

        if op == "PUSH_CONST":
            stack.append(args[0])
        elif op == "LOAD_VAR":
            stack.append(env.get(args[0], 0))
        elif op == "STORE_VAR":
            val = pop()
            env[args[0]] = val
        elif op == "NEG":
            val = pop()
            stack.append(-val)
        elif op == "ADD":
            b = pop()
            a = pop()
            stack.append(a + b)
        # ... other ops (SUB, MUL, DIV, CMP_*, PRINT, JUMP, JUMP_IF_FALSE, LABEL)
        pc += 1

    return env
```

**Demo programs**

```python
DEMO1 = """x = 1 + 2 * 3
y = x + 4
print(y)
"""

DEMO2 = """x = 0
while x < 5:
    x = x + 1
print(x)
"""

DEMO3 = """x = 10
if x > 5:
    print(x)
else:
    print(0)
"""
```

```python
def compile_and_run(source: str, name: str = "demo"):
    print("=" * 60)
    print(f"Source: {name}")
    print("-" * 60)
    print(source.rstrip())
    print("
[1] Tokens")
    toks = lex(source)
    print(format_tokens(toks))

    print("
[2] Parse / AST")
    parser = Parser(toks)
    prog = parser.parse()
    if parser.errors:
        print("Parser errors:")
        for e in parser.errors:
            print("  -", e)
        print("No code generated due to errors.")
        return
    print(format_ast(prog))

    print("
[3] Generated Assembly")
    code = generate_program(prog)
    print(format_code(code))

    print("
[4] VM Execution Output & Final Variables")
    env = run_program(code)
    print("Final env:", env)
    print("=" * 60)
    print()
```

In `__main__`, you call:

```python
compile_and_run(DEMO1, "Demo 1: Arithmetic & Print")
compile_and_run(DEMO2, "Demo 2: While Loop")
compile_and_run(DEMO3, "Demo 3: If/Else")
```

---

## 9. Error Handling: Line Numbers & Messages

**What this proves**

- Your compiler **detects syntax errors**, reports **line numbers**, and **stops code generation** when errors exist.

**Key code (error tracking in Parser)**

```python
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.errors: List[str] = []

    def expect(self, *types) -> Token:
        tok = self.current()
        if tok.type in types:
            return self.advance()
        msg = f"Expected {types}, got {tok.type} at line {tok.line}"
        self.errors.append(msg)
        # simple recovery: skip until NEWLINE or DEDENT or EOF
        while self.current().type not in {"NEWLINE", "DEDENT", "EOF"}:
            self.advance()
        return tok
```

Error demo program:

```python
ERROR_DEMO = """x = 1 +
if x > :
    print(x)
"""
```

Used in:

```python
compile_and_run(ERROR_DEMO, "Error Demo: Syntax Errors")
```

If `parser.errors` is non-empty:

```python
if parser.errors:
    print("Parser errors:")
    for e in parser.errors:
        print("  -", e)
    print("No code generated due to errors.")
    return
```

---

## 10. Dataflow & Memory Management

**What this proves**

- You have a clear **memory model** (environment / symbol table).
- You can discuss **dataflow** (values flowing from literals → stack → variables).

**Key code (environment & stack)**

```python
stack: List[Any] = []
env: Dict[str, Any] = {}
```

```python
elif op == "STORE_VAR":
    val = pop()
    env[args[0]] = val
```

```python
elif op == "LOAD_VAR":
    stack.append(env.get(args[0], 0))
```

You can explain:

- `env` is your **symbol table / memory** for variables.  
- `stack` is where **expression evaluation** happens (dataflow).  

---

## 11. Limitations & How to Run

**How to run (terminal command)**

You can show this on the slide:

```bash
python main.py
```

**Entry point code**

```python
if __name__ == "__main__":
    print("Mini-Python Subset Compiler Demo\n")
    show_theory()
    compile_and_run(DEMO1, "Demo 1: Arithmetic & Print")
    compile_and_run(DEMO2, "Demo 2: While Loop")
    compile_and_run(DEMO3, "Demo 3: If/Else")
    compile_and_run(ERROR_DEMO, "Error Demo: Syntax Errors")
```

**Limitations** (you describe in bullets on slide, not in code):

- Only a **subset of Python** (assignments, if/else, while, print, ints).  
- No functions, lists, dictionaries, or advanced types.  
- Assembly is a simple **custom stack machine**, not real CPU ISA.  
- No advanced optimizations (only straightforward code generation).

---

Use this `description.md` as your **cheat sheet**:  
Each section = one or more slides, and each code block = a snippet you can screenshot or paste into your presentation.
