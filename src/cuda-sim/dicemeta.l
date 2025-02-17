%option nounput
%option noyywrap
%option yylineno
%option prefix="dicemeta_"

%option bison-bridge
%option reentrant

%{
#include "opcodes.h"
#include "dice_metadata.h"
#include "dicemeta.tab.h"
#include <string.h>
#include <stdlib.h>
#include "../../libcuda/gpgpu_context.h"

#define LINEBUF_SIZE (4*1024)
#define TC dicemeta_parser->col+=strlen(yytext);
#define CHECK_UNSIGNED \
	if( yytext[strlen(yytext)-1]=='U' ) { \
		printf("DICE: ERROR ** U modifier not implemented\n"); \
		abort(); \
	}
#define YY_DECL int dicemeta_lex \
	       (YYSTYPE * yylval_param , yyscan_t yyscanner, dice_metadata_parser* dicemeta_parser)
int dicemeta_error( yyscan_t yyscanner, dice_metadata_parser* dicemeta_parser, const char *s );
%}

/*----------------------------------------------------------------------------
  Flex rules return tokens for keywords, numbers, punctuation and register
  operands (which begin with '%' or '$').  Whitespace is ignored.
----------------------------------------------------------------------------*/
%%

"DBB_ID" TC; return DBB_ID; 
"UNROLLING_FACTOR" TC; return UNROLLING_FACTOR; 
"UNROLLING_STRATEGY" TC; return UNROLLING_STRATEGY;
"LAT" TC; return LAT; 
"IN_REGS" TC; return IN_REGS; 
"OUT_REGS" TC; return OUT_REGS; 
"LD_DEST_REGS" TC; return LD_DEST_REGS; 
"STORE" TC; return STORE; 
"BRANCH_UNI" TC; return BRANCH_UNI; 
"BRANCH_PRED" TC; return BRANCH_PRED; 
"BRANCH_TARGET" TC; return BRANCH_TARGET; 
"BRANCH_RECVPC" TC; return BRANCH_RECVPC; 
"BRANCH" TC; return BRANCH; 
"RET" TC; return RET; 
"BITSTREAM_ADDR" TC; return BITSTREAM_ADDR;
"BITSTREAM_LENGTH" TC; return BITSTREAM_LENGTH;
"FUNCTION" TC; return FUNCTION;



"%clock" TC; yylval->int_value = CLOCK_REG; return SPECIAL_REGISTER;
"%halfclock" TC; yylval->int_value = HALFCLOCK_ID; return SPECIAL_REGISTER;
"%clock64" TC; yylval->int_value = CLOCK64_REG; return SPECIAL_REGISTER;
"%ctaid" TC; yylval->int_value = CTAID_REG; return SPECIAL_REGISTER;
"%envreg"[0-9]+ TC; sscanf(yytext+7,"%u",&yylval->int_value); yylval->int_value<<=16; yylval->int_value += ENVREG_REG; return SPECIAL_REGISTER;
"%gridid" TC; yylval->int_value = GRIDID_REG; return SPECIAL_REGISTER;
"%laneid"  TC; yylval->int_value = LANEID_REG; return SPECIAL_REGISTER;
"%lanemask_eq"  TC; yylval->int_value = LANEMASK_EQ_REG; return SPECIAL_REGISTER;
"%lanemask_le"  TC; yylval->int_value = LANEMASK_LE_REG; return SPECIAL_REGISTER;
"%lanemask_lt"  TC; yylval->int_value = LANEMASK_LT_REG; return SPECIAL_REGISTER;
"%lanemask_ge"  TC; yylval->int_value = LANEMASK_GE_REG; return SPECIAL_REGISTER;
"%lanemask_gt"  TC; yylval->int_value = LANEMASK_GT_REG; return SPECIAL_REGISTER;
"%nctaid" TC; yylval->int_value = NCTAID_REG; return SPECIAL_REGISTER;
"%ntid"  TC; yylval->int_value = NTID_REG; return SPECIAL_REGISTER;
"%nsmid"  TC; yylval->int_value = NSMID_REG; return SPECIAL_REGISTER;
"%nwarpid"  TC; yylval->int_value = NWARPID_REG; return SPECIAL_REGISTER;
"%pm"[0-3]  TC; sscanf(yytext+3,"%u",&yylval->int_value); yylval->int_value<<=16; yylval->int_value += PM_REG; return SPECIAL_REGISTER;
"%smid"  TC; yylval->int_value = SMID_REG; return SPECIAL_REGISTER;
"%tid"  TC; yylval->int_value = TID_REG; return SPECIAL_REGISTER;
"%warpid"  TC; yylval->int_value = WARPID_REG; return SPECIAL_REGISTER;
"WARP_SZ"  TC; yylval->int_value = WARPSZ_REG; return SPECIAL_REGISTER;

[a-zA-Z_][a-zA-Z0-9_$]*  TC; yylval->string_value = strdup(yytext); return FUNCTION_IDENTIFIER;
[%][a-zA-Z0-9_$]+  TC; yylval->string_value = strdup(yytext); return REGOPERAND;
[$][a-zA-Z0-9_]+  TC; yylval->string_value = strdup(yytext); return LABEL;

[0-9]+  TC; yylval->int_value = atoi(yytext); return NUMBER;
\.x	TC; yylval->int_value = 0; return DIMENSION_MODIFIER;
\.y	TC; yylval->int_value = 1; return DIMENSION_MODIFIER;
\.z	TC; yylval->int_value = 2; return DIMENSION_MODIFIER;

","     TC; return COMMA;
"("	TC; return LEFT_PAREN;
")"	TC; return RIGHT_PAREN;
";"	TC; return SEMI_COLON;
"!"	TC; return EXCLAMATION;
"=" 	TC; return EQUALS;

"//"[^\n]* TC;	// eat single

\n.*  dicemeta_parser->col=0; strncpy(dicemeta_parser->linebuf, yytext + 1, LINEBUF_SIZE); yyless( 1 );

" " TC;
"\t" TC;

%%
int dicemeta_error( yyscan_t yyscanner, dice_metadata_parser* dicemeta_parser, const char *s )
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
	int i;
	dicemeta_parser->g_error_detected = 1;
	fflush(stdout);
	if( s != NULL )
		printf("%s:%u Syntax error:\n\n", dicemeta_parser->g_dice_metadata_filename, yylineno );
	printf("   %s\n", dicemeta_parser->linebuf );
	printf("   ");
	for( i=0; i < dicemeta_parser->col-1; i++ ) {
		if( dicemeta_parser->linebuf[i] == '\t' ) printf("\t");
		else printf(" ");
	}
			
	printf("^\n\n");
	fflush(stdout);
	exit(1);
	return 0;
}
