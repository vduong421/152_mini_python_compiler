#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mini-Python Subset Compiler -> Custom Stack Assembly

Features:
- Lexical analysis (tokens + regex)
- Indentation-based blocks (subset of Python)
- Recursive descent parser building an AST
- Simple custom assembly code generation
- Tiny VM to execute the custom assembly
- Demo test cases + error handling
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Tuple

# -----------------------------
# 1. TOKENS & LEXER  (Lexical analysis, regex, token classification)
# -----------------------------

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


@dataclass
class Token:
    type: str
    value: Any
    line: int
    column: int


class LexerError(Exception):
    pass


def lex(source: str) -> List[Token]:
    tokens: List[Token] = []
    indent_stack = [0]

    lines = source.splitlines()
    for lineno, raw_line in enumerate(lines, start=1):
        # Handle indentation (count spaces at start)
        stripped = raw_line.lstrip(" ")
        if stripped == "" or stripped.startswith("#"):
            # Blank or comment-only line -> emit NEWLINE only, no indent change
            tokens.append(Token("NEWLINE", "\\n", lineno, 0))
            continue

        indent = len(raw_line) - len(stripped)
        if indent % 4 != 0:
            # non-multiple of 4 indentation: you could log an indentation warning here
            pass

        if indent > indent_stack[-1]:
            indent_stack.append(indent)
            tokens.append(Token("INDENT", indent, lineno, 0))
        else:
            while indent < indent_stack[-1]:
                indent_stack.pop()
                tokens.append(Token("DEDENT", indent, lineno, 0))
            if indent != indent_stack[-1]:
                # inconsistent indentation - treat as error token but continue
                tokens.append(Token("INDENT_ERROR", indent, lineno, 0))

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
                # Ignore rest of line
                break
            elif kind == "MISMATCH":
                raise LexerError(f"Unexpected character {value!r} at line {lineno}, col {col}")
            pos = mo.end()
        tokens.append(Token("NEWLINE", "\\n", lineno, len(raw_line) + 1))

    # Close any remaining indents
    lineno = len(lines) + 1
    while len(indent_stack) > 1:
        indent_stack.pop()
        tokens.append(Token("DEDENT", 0, lineno, 0))
    tokens.append(Token("EOF", "", lineno, 0))
    return tokens

# -----------------------------
# 2. AST NODES (Parse tree / AST representation)
# -----------------------------

class AST:  # base class
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

# -----------------------------
# 3. PARSER (CFG, left/right derivation, syntax errors, AST build)
# -----------------------------

class ParserError(Exception):
    pass


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.errors: List[str] = []

    def current(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def match(self, *types) -> Optional[Token]:
        if self.current().type in types:
            return self.advance()
        return None

    def expect(self, *types) -> Token:
        tok = self.current()
        if tok.type in types:
            return self.advance()
        msg = f"Expected {types}, got {tok.type} at line {tok.line}"
        self.errors.append(msg)
        # Attempt simple recovery: skip until NEWLINE or DEDENT or EOF
        while self.current().type not in {"NEWLINE", "DEDENT", "EOF"}:
            self.advance()
        return tok  # return offending token so parsing can continue

    # --------- Entry points ----------

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
            # assignment?
            if self.tokens[self.pos + 1].type == "ASSIGN":
                return self.parse_assign()
            else:
                msg = f"Unexpected identifier (not assignment) at line {tok.line}"
                self.errors.append(msg)
                while self.current().type not in {"NEWLINE", "EOF"}:
                    self.advance()
                self.match("NEWLINE")
                return None
        else:
            msg = f"Unexpected token {tok.type} at line {tok.line}"
            self.errors.append(msg)
            while self.current().type not in {"NEWLINE", "EOF"}:
                self.advance()
            self.match("NEWLINE")
            return None

    def parse_block(self) -> List[AST]:
        # we expect an INDENT followed by statements, ending with DEDENT
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

    def parse_if(self) -> If:
        tok_if = self.expect("IF")
        cond = self.parse_expression()
        self.expect("COLON")
        self.expect("NEWLINE")
        then_body = self.parse_block()
        else_body = None
        # optional else
        if self.current().type == "ELSE":
            self.advance()
            self.expect("COLON")
            self.expect("NEWLINE")
            else_body = self.parse_block()
        return If(cond, then_body, else_body, tok_if.line)

    def parse_while(self) -> While:
        tok_w = self.expect("WHILE")
        cond = self.parse_expression()
        self.expect("COLON")
        self.expect("NEWLINE")
        body = self.parse_block()
        return While(cond, body, tok_w.line)

    def parse_assign(self) -> Assign:
        name_tok = self.expect("ID")
        self.expect("ASSIGN")
        expr = self.parse_expression()
        self.match("NEWLINE")
        return Assign(name_tok.value, expr, name_tok.line)

    def parse_print(self) -> Print:
        tok_p = self.expect("PRINT")
        self.expect("LPAREN")
        expr = self.parse_expression()
        self.expect("RPAREN")
        self.match("NEWLINE")
        return Print(expr, tok_p.line)

    # --------- Expressions (recursive descent) ----------

    def parse_expression(self) -> AST:
        return self.parse_equality()

    def parse_equality(self) -> AST:
        node = self.parse_comparison()
        while self.current().type in {"EQ", "NE"}:
            op_tok = self.advance()
            right = self.parse_comparison()
            node = BinOp(op_tok.type, node, right, op_tok.line)
        return node

    def parse_comparison(self) -> AST:
        node = self.parse_term()
        while self.current().type in {"LT", "LE", "GT", "GE"}:
            op_tok = self.advance()
            right = self.parse_term()
            node = BinOp(op_tok.type, node, right, op_tok.line)
        return node

    def parse_term(self) -> AST:
        node = self.parse_factor()
        while self.current().type in {"PLUS", "MINUS"}:
            op_tok = self.advance()
            right = self.parse_factor()
            node = BinOp(op_tok.type, node, right, op_tok.line)
        return node

    def parse_factor(self) -> AST:
        node = self.parse_unary()
        while self.current().type in {"MUL", "DIV"}:
            op_tok = self.advance()
            right = self.parse_unary()
            node = BinOp(op_tok.type, node, right, op_tok.line)
        return node

    def parse_unary(self) -> AST:
        if self.current().type in {"PLUS", "MINUS"}:
            op_tok = self.advance()
            operand = self.parse_unary()
            return UnaryOp(op_tok.type, operand, op_tok.line)
        return self.parse_primary()

    def parse_primary(self) -> AST:
        tok = self.current()
        if tok.type == "NUMBER":
            self.advance()
            return Number(tok.value, tok.line)
        elif tok.type == "ID":
            self.advance()
            return Var(tok.value, tok.line)
        elif tok.type == "LPAREN":
            self.advance()
            expr = self.parse_expression()
            self.expect("RPAREN")
            return expr
        else:
            msg = f"Unexpected token {tok.type} in expression at line {tok.line}"
            self.errors.append(msg)
            self.advance()
            return Number(0, tok.line)

# -----------------------------
# 4. CODE GENERATION (Custom assembly)
# -----------------------------

@dataclass
class Instr:
    op: str
    args: Tuple[Any, ...] = ()


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
            elif node.op == "PLUS":
                pass
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
            asm_op = op_map.get(node.op)
            if not asm_op:
                raise ValueError(f"Unknown binary op {node.op}")
            code.append(Instr(asm_op, ()))
        else:
            raise TypeError(f"Unsupported expr node: {node}")

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
        else:
            raise TypeError(f"Unsupported stmt node: {node}")

    for stmt in prog.body:
        gen_stmt(stmt)
    return code

# -----------------------------
# 5. SIMPLE VM TO RUN ASSEMBLY (Result of compilation)
# -----------------------------

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
        elif op == "SUB":
            b = pop()
            a = pop()
            stack.append(a - b)
        elif op == "MUL":
            b = pop()
            a = pop()
            stack.append(a * b)
        elif op == "DIV":
            b = pop()
            a = pop()
            stack.append(a // b)
        elif op == "CMP_LT":
            b = pop()
            a = pop()
            stack.append(a < b)
        elif op == "CMP_LE":
            b = pop()
            a = pop()
            stack.append(a <= b)
        elif op == "CMP_GT":
            b = pop()
            a = pop()
            stack.append(a > b)
        elif op == "CMP_GE":
            b = pop()
            a = pop()
            stack.append(a >= b)
        elif op == "CMP_EQ":
            b = pop()
            a = pop()
            stack.append(a == b)
        elif op == "CMP_NE":
            b = pop()
            a = pop()
            stack.append(a != b)
        elif op == "PRINT":
            val = pop()
            print(f"[VM PRINT] {val}")
        elif op == "JUMP":
            pc = args[0]
            continue
        elif op == "JUMP_IF_FALSE":
            cond = pop()
            if not cond:
                pc = args[0]
                continue
        elif op == "LABEL":
            pc += 1
            continue
        else:
            raise RuntimeError(f"Unknown instruction {op}")

        pc += 1

    return env

# -----------------------------
# 6. PRETTY-PRINT HELPERS (For slides: tokens, AST, code)
# -----------------------------

def format_tokens(tokens: List[Token]) -> str:
    parts = []
    for t in tokens:
        if t.type in {"NEWLINE"}:
            parts.append(f"(NEWLINE)")
        elif t.type in {"INDENT", "DEDENT"}:
            parts.append(f"({t.type})")
        else:
            parts.append(f"({t.type}:{t.value})")
    return " ".join(parts)

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
    if isinstance(node, UnaryOp):
        lines = [pad + f"UnaryOp({node.op})"]
        lines.append(format_ast(node.operand, indent + 1))
        return "\n".join(lines)
    if isinstance(node, Assign):
        lines = [pad + f"Assign({node.name})"]
        lines.append(format_ast(node.expr, indent + 1))
        return "\n".join(lines)
    if isinstance(node, Print):
        lines = [pad + "Print"]
        lines.append(format_ast(node.expr, indent + 1))
        return "\n".join(lines)
    if isinstance(node, If):
        lines = [pad + "If"]
        lines.append(pad + "  Cond:")
        lines.append(format_ast(node.cond, indent + 2))
        lines.append(pad + "  Then:")
        for s in node.then_body:
            lines.append(format_ast(s, indent + 2))
        if node.else_body is not None:
            lines.append(pad + "  Else:")
            for s in node.else_body:
                lines.append(format_ast(s, indent + 2))
        return "\n".join(lines)
    if isinstance(node, While):
        lines = [pad + "While"]
        lines.append(pad + "  Cond:")
        lines.append(format_ast(node.cond, indent + 2))
        lines.append(pad + "  Body:")
        for s in node.body:
            lines.append(format_ast(s, indent + 2))
        return "\n".join(lines)
    return pad + repr(node)

def format_code(code: List[Instr]) -> str:
    lines = []
    for i, instr in enumerate(code):
        lines.append(f"{i:03}: {instr.op} {', '.join(map(str, instr.args))}")
    return "\n".join(lines)

# -----------------------------
# 7. THEORY STRINGS (Regex, CFG, CNF, GNF, DFA) – for theory slides
# -----------------------------

REGEX_SPEC_TEXT = r"""
Identifier (ID):   [A-Za-z_][A-Za-z0-9_]*
Number (NUMBER):   [0-9]+
Whitespace:        [ \t]+
Comments:          \#.*
Keywords:          if | else | while | print
Operators:         == != <= >= < > + - * /
"""

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

CNF_SNIPPET = r"""
Example CNF for a tiny subset:

S  -> IF_COND
IF_COND -> IF_EXPR COLON_BLOCK
IF_EXPR -> IF Expr
COLON_BLOCK -> COLON BLOCK
BLOCK -> NEWLINE INDENT StmtList DEDENT

(All productions have at most 2 non-terminals or 1 terminal.)
"""

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

# -----------------------------
# 8. DEMO PROGRAMS (3 good + 1 error case)
# -----------------------------

DEMO1 = """\
x = 1 + 2 * 3
y = x + 4
print(y)
"""

DEMO2 = """\
x = 0
while x < 5:
    x = x + 1
print(x)
"""

DEMO3 = """\
x = 10
if x > 5:
    print(x)
else:
    print(0)
"""

ERROR_DEMO = """\
x = 1 +
if x > :
    print(x)
"""

def compile_and_run(source: str, name: str = "demo"):
    print("=" * 60)
    print(f"Source: {name}")
    print("-" * 60)
    print(source.rstrip())
    print("\n[1] Tokens")
    toks = lex(source)
    print(format_tokens(toks))

    print("\n[2] Parse / AST")
    parser = Parser(toks)
    prog = parser.parse()
    if parser.errors:
        print("Parser errors:")
        for e in parser.errors:
            print("  -", e)
        print("No code generated due to errors.")
        return
    print(format_ast(prog))

    print("\n[3] Generated Assembly")
    code = generate_program(prog)
    print(format_code(code))

    print("\n[4] VM Execution Output & Final Variables")
    env = run_program(code)
    print("Final env:", env)
    print("=" * 60)
    print()

def show_theory():
    print("=" * 60)
    print("LEXICAL REGEX SPEC")
    print("=" * 60)
    print(REGEX_SPEC_TEXT)
    print("=" * 60)
    print("GRAMMAR (CFG)")
    print("=" * 60)
    print(GRAMMAR_TEXT)
    print("=" * 60)
    print("CNF SNIPPET")
    print("=" * 60)
    print(CNF_SNIPPET)
    print("=" * 60)
    print("GNF SNIPPET")
    print("=" * 60)
    print(GNF_SNIPPET)
    print("=" * 60)
    print("DFA DESCRIPTION")
    print("=" * 60)
    print(DFA_DESCRIPTION)
    print("=" * 60)

if __name__ == "__main__":
    print("Mini-Python Subset Compiler Demo\n")
    show_theory()
    compile_and_run(DEMO1, "Demo 1: Arithmetic & Print")
    compile_and_run(DEMO2, "Demo 2: While Loop")
    compile_and_run(DEMO3, "Demo 3: If/Else")
    compile_and_run(ERROR_DEMO, "Error Demo: Syntax Errors")
