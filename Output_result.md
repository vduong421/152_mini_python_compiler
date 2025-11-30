PS G:\_Code\& School Code\152_mini_python_compiler> python main.py
Mini Python Compiler

============================================================
LEXICAL REGEX SPEC
============================================================

Identifier (ID):   [A-Za-z_][A-Za-z0-9_]*
Number (NUMBER):   [0-9]+
Whitespace:        [ \t]+
Comments:          \#.*
Keywords:          if | else | while | print
Operators:         == != <= >= < > + - * /

============================================================
GRAMMAR (CFG)
============================================================

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

============================================================
CNF SNIPPET
============================================================

Example CNF for a tiny subset:

S  -> IF_COND
IF_COND -> IF_EXPR COLON_BLOCK
IF_EXPR -> IF Expr
COLON_BLOCK -> COLON BLOCK
BLOCK -> NEWLINE INDENT StmtList DEDENT

(All productions have at most 2 non-terminals or 1 terminal.)

============================================================
GNF SNIPPET
============================================================

Example GNF for expression grammar subset:

Expr -> NUMBER ExprTail
      | ID ExprTail
ExprTail -> PLUS Expr
          | MINUS Expr
          | EQ Expr
          | ε

(Each production starts with a terminal, followed by non-terminals.)

============================================================
DFA DESCRIPTION
============================================================

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

============================================================
============================================================
Source: Demo 1: Arithmetic & Print
------------------------------------------------------------
x = 1 + 2 * 3
y = x + 4
print(y)

[1] Tokens
(ID:x) (ASSIGN:=) (NUMBER:1) (PLUS:+) (NUMBER:2) (MUL:*) (NUMBER:3) (NEWLINE) (ID:y) (ASSIGN:=) (ID:x) (PLUS:+) (NUMBER:4) (NEWLINE) (PRINT:print) (LPAREN:() (ID:y) (RPAREN:)) (NEWLINE) (EOF:)

[2] Parse / AST
Program
  Assign(x)
    BinOp(PLUS)
      Number(1)
      BinOp(MUL)
        Number(2)
        Number(3)
  Assign(y)
    BinOp(PLUS)
      Var(x)
      Number(4)
  Print
    Var(y)

[3] Generated Assembly (x86)
section .data
    x dd 0
    y dd 0

section .text
    global _start
_start:
    mov eax, 1
    push eax
    mov eax, 2
    push eax
    mov eax, 3
    pop ebx
    imul eax, ebx
    pop ebx
    add eax, ebx
    mov dword [x], eax
    mov eax, dword [x]
    push eax
    mov eax, 4
    pop ebx
    add eax, ebx
    mov dword [y], eax
    mov eax, dword [y]
    ; print value in eax (pseudo-call)
    ; push eax
    ; call print_int
    ; add esp, 4

    ; exit(0)
    mov eax, 1
    mov ebx, 0
    int 0x80

[4] VM Execution Output & Final Variables
[VM PRINT] 11
Final env: {'x': 7, 'y': 11}
============================================================

============================================================
Source: Demo 2: While Loop
------------------------------------------------------------
x = 0
while x < 5:
    x = x + 1
print(x)

[1] Tokens
(ID:x) (ASSIGN:=) (NUMBER:0) (NEWLINE) (WHILE:while) (ID:x) (LT:<) (NUMBER:5) (COLON::) (NEWLINE) (INDENT) (ID:x) (ASSIGN:=) (ID:x) (PLUS:+) (NUMBER:1) (NEWLINE) 
(DEDENT) (PRINT:print) (LPAREN:() (ID:x) (RPAREN:)) (NEWLINE) (EOF:)

[2] Parse / AST
Program
  Assign(x)
    Number(0)
  While
    Cond:
      BinOp(LT)
        Var(x)
        Number(5)
    Body:
      Assign(x)
        BinOp(PLUS)
          Var(x)
          Number(1)
  Print
    Var(x)

[3] Generated Assembly (x86)
section .data
    x dd 0

section .text
    global _start
_start:
    mov eax, 0
    mov dword [x], eax
.Lloop0:
    mov eax, dword [x]
    push eax
    mov eax, 5
    pop ebx
    cmp ebx, eax
    setl al
    movzx eax, al
    cmp eax, 0
    je .Lend1
    mov eax, dword [x]
    push eax
    mov eax, 1
    pop ebx
    add eax, ebx
    mov dword [x], eax
    jmp .Lloop0
.Lend1:
    mov eax, dword [x]
    ; print value in eax (pseudo-call)
    ; push eax
    ; call print_int
    ; add esp, 4

    ; exit(0)
    mov eax, 1
    mov ebx, 0
    int 0x80

[4] VM Execution Output & Final Variables
[VM PRINT] 5
Final env: {'x': 5}
============================================================

============================================================
Source: Demo 3: If/Else
------------------------------------------------------------
x = 10
if x > 5:
    print(x)
else:
    print(0)

[1] Tokens
(ID:x) (ASSIGN:=) (NUMBER:10) (NEWLINE) (IF:if) (ID:x) (GT:>) (NUMBER:5) (COLON::) (NEWLINE) (INDENT) (PRINT:print) (LPAREN:() (ID:x) (RPAREN:)) (NEWLINE) (DEDENT) (ELSE:else) (COLON::) (NEWLINE) (INDENT) (PRINT:print) (LPAREN:() (NUMBER:0) (RPAREN:)) (NEWLINE) (DEDENT) (EOF:)

[2] Parse / AST
Program
  Assign(x)
    Number(10)
  If
    Cond:
      BinOp(GT)
        Var(x)
        Number(5)
    Then:
      Print
        Var(x)
    Else:
      Print
        Number(0)

[3] Generated Assembly (x86)
section .data
    x dd 0

section .text
    global _start
_start:
    mov eax, 10
    mov dword [x], eax
    mov eax, dword [x]
    push eax
    mov eax, 5
    pop ebx
    cmp ebx, eax
    setg al
    movzx eax, al
    cmp eax, 0
    je .Lelse0
    mov eax, dword [x]
    ; print value in eax (pseudo-call)
    ; push eax
    ; call print_int
    ; add esp, 4
    jmp .Lend1
.Lelse0:
    mov eax, 0
    ; print value in eax (pseudo-call)
    ; push eax
    ; call print_int
    ; add esp, 4
.Lend1:

    ; exit(0)
    mov eax, 1
    mov ebx, 0
    int 0x80

[4] VM Execution Output & Final Variables
[4] VM Execution Output & Final Variables
[VM PRINT] 10
Final env: {'x': 10}
============================================================

============================================================
Source: Error Demo: Syntax Errors
------------------------------------------------------------
x = 1 +
if x > :
    print(x)

[1] Tokens
(ID:x) (ASSIGN:=) (NUMBER:1) (PLUS:+) (NEWLINE)
(IF:if) (ID:x) (GT:>) (COLON::) (NEWLINE)
(INDENT)
(PRINT:print) (LPAREN:() (ID:x) (RPAREN:)) (NEWLINE)
(DEDENT)
(EOF:)

[2] Parse / AST
Parser errors:
  - Unexpected token NEWLINE in expression at line 1
  - Unexpected token COLON in expression at line 2
  - Expected ('COLON',), got NEWLINE at line 2
No code generated due to errors.
PS G:\_Code\& School Code\152_mini_python_compiler>


