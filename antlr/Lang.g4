// Lang.g4

grammar Lang;

// --- Rules (Parser Rules) ---

program: statement* EOF;

statement:
	varDecl SEMI				# VarDeclStmt
	| functionDecl				# FuncDeclStmt
	| ifStatement				# IfStmt
	| whileStatement			# WhileStmt
	| printStatement SEMI		# PrintStmt
	| returnStatement SEMI		# ReturnStmt
	| BREAK SEMI				# BreakStmt
	| CONTINUE SEMI				# ContinueStmt
	| blockStatement			# BlockStmt
	| assignmentStatement SEMI	# AssignStmt
	| expression SEMI			# ExprStmt;

blockStatement: LBRACE statement* RBRACE # Block;

varDecl: (LET | CONST) ID ASSIGN expression;

functionDecl: FN ID LPAREN parameterList? RPAREN blockStatement;

ifStatement:
	IF LPAREN expression RPAREN statement (ELSE statement)?;

whileStatement: WHILE LPAREN expression RPAREN statement;

printStatement: PRINT LPAREN expressionList RPAREN;

returnStatement: RETURN expression?;

assignmentStatement:
	ID (
		ASSIGN
		| ADD_ASSIGN
		| SUB_ASSIGN
		| MUL_ASSIGN
		| DIV_ASSIGN
	) expression;

parameterList: ID (COMMA ID)*;

expressionList: expression (COMMA expression)*;

// --- Expressions (Expression Rules) ---

expression: ternaryExpression;

ternaryExpression:
	logicalOr (QUESTION expression COLON expression)?;

logicalOr: logicalAnd (OR logicalAnd)*;

logicalAnd: equality (AND equality)*;

equality: relational ((EQ | NEQ) relational)*;

relational: additive ((LT | GT | LE | GE) additive)*;

additive: multiplicative ((PLUS | MINUS) multiplicative)*;

multiplicative: unary ((MUL | DIV | MOD) unary)*;

unary: (MINUS | NOT | DEC) unary	# UnaryOp
	| TYPEOF unary					# TypeofOp
	| postfix						# PostfixExpr;

postfix:
	primary (LPAREN expressionList? RPAREN | DEC)* # FunctionCall;

primary:
	INT														# IntLiteral
	| DOUBLE												# DoubleLiteral
	| STRING												# StringLiteral
	| TRUE													# BoolTrue
	| FALSE													# BoolFalse
	| ID													# Variable
	| LPAREN expression RPAREN								# Parenthesized
	| FN LPAREN parameterList? RPAREN ARROW blockStatement	# AnonymousFunction;

// --- Lexical Tokens (Lexer Rules) ---

// Keywords
LET: 'let';
CONST: 'const';
FN: 'fn';
RETURN: 'return';
IF: 'if';
ELSE: 'else';
WHILE: 'while';
BREAK: 'break';
CONTINUE: 'continue';
PRINT: 'print';
TYPEOF: 'typeof';
TRUE: 'true';
FALSE: 'false';

// Identifiers
ID: [a-zA-Z_] [a-zA-Z_0-9]*;

// Literals
INT: [0-9]+;
DOUBLE: [0-9]+ '.' [0-9]+;
STRING: '"' (~["\r\n])* '"';

// Operators
PLUS: '+';
MINUS: '-';
MUL: '*';
DIV: '/';
MOD: '%';
DEC: '--';
EQ: '==';
NEQ: '!=';
LT: '<';
GT: '>';
LE: '<=';
GE: '>=';
AND: '&&';
OR: '||';
NOT: '!';
ASSIGN: '=';
ADD_ASSIGN: '+=';
SUB_ASSIGN: '-=';
MUL_ASSIGN: '*=';
DIV_ASSIGN: '/=';
ARROW: '=>';
QUESTION: '?';
COLON: ':';

// Delimiters
LPAREN: '(';
RPAREN: ')';
LBRACE: '{';
RBRACE: '}';
SEMI: ';';
COMMA: ',';

// Ignored Characters
WS: [ \t\r\n]+ -> skip;
COMMENT: '//' .*? '\n' -> skip;