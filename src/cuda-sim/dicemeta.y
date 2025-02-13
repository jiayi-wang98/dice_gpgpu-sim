%{
typedef void * yyscan_t;
class dice_metadata_parser;
#include "../../libcuda/gpgpu_context.h"

static const char* special_reg_names[] = {
    "CLOCK_REG",
    "HALFCLOCK_ID",
    "CLOCK64_REG",
    "CTAID_REG",
    "ENVREG_REG",
    "GRIDID_REG",
    "LANEID_REG",
    "LANEMASK_EQ_REG",
    "LANEMASK_LE_REG",
    "LANEMASK_LT_REG",
    "LANEMASK_GE_REG",
    "LANEMASK_GT_REG",
    "NCTAID_REG",
    "NTID_REG",
    "NSMID_REG",
    "NWARPID_REG",
    "PM_REG",
    "SMID_REG",
    "TID_REG",
    "WARPID_REG",
    "WARPSZ_REG"
};

void yyerror(dice_metadata_parser *dicemeta_parser, const char *s) {
    fprintf(stderr, "Parse error: %s\n", s);
}
%}

%define api.pure full
%parse-param {yyscan_t scanner}
%parse-param {dice_metadata_parser* dicemeta_parser}
%lex-param {yyscan_t scanner}
%lex-param {dice_metadata_parser* dicemeta_parser}

/*----------------------------------------------------------------------------
  The semantic value union supports:
    - an integer (for NUMBER),
    - a string (for REGOPERAND), and
    - a pointer to a std::list<operand_info> (for register lists).
----------------------------------------------------------------------------*/
%union {
  double double_value;
  float  float_value;
  int    int_value;
  char * string_value;
  const char * const_string_value;
  void * ptr_value;
}

/*----------------------------------------------------------------------------
  Declare tokens.
----------------------------------------------------------------------------*/
%token DBB_ID UNROLLING_FACTOR UNROLLING_STRATEGY LAT IN_REGS OUT_REGS LD_DEST_REGS STORE 
%token BRANCH BRANCH_UNI BRANCH_PRED BRANCH_TARGET BRANCH_RECVPC RET
%token <int_value> NUMBER
%token <int_value> SPECIAL_REGISTER
%token <string_value> REGOPERAND
%token <int_value> DIMENSION_MODIFIER
%token COMMA LEFT_PAREN RIGHT_PAREN SEMI_COLON EXCLAMATION EQUALS

%type <const_string_value> operands

%{
  	#include "ptx_parser.h"
	#include <stdlib.h>
	#include <string.h>
	#include <math.h>
	void syntax_not_implemented(yyscan_t yyscanner, dice_metadata_parser* dicemeta_parser);
	int dicemeta_lex(YYSTYPE * yylval_param, yyscan_t yyscanner, dice_metadata_parser* dicemeta_parser);
	int dicemeta_error( yyscan_t yyscanner, dice_metadata_parser* dicemeta_parser, const char *s );
%}

%%

/* A meta file is simply a series of blocks */
metadata_file:
      /* empty */
    | metadata_file block
    ;

/* 
   Each block starts with a DBB_ID field and ends with a ';'. When the block is complete,
   the just‐filled dice_metadata (pointed to by current_metadata) is added to the parser list.
*/
block:
    dbbid_field opt_fields SEMI_COLON {
        printf("DICE METADATA PARSER: Finished current DBB.\n"); fflush(stdout);
        dicemeta_parser->commit_dbb();
    }
    ;

/* 
   The DBB_ID field is mandatory and creates the dice_metadata object.
*/
dbbid_field:
    DBB_ID EQUALS NUMBER COMMA{
         printf("DICE METADATA PARSER: DBB_ID = %d\n", $3); fflush(stdout);
         dicemeta_parser->create_new_dbb($3);
    }
    ;

/* There may be additional fields after DBB_ID. */
opt_fields:
    field_list
    ;

/* Fields are allowed to appear separated by commas or by whitespace. */
field_list:
      field
    | field_list COMMA field
    | field_list field
    ;

/*----------------------------------------------------------------------------
  Recognize each possible field.
  
  Note:
   - The unrolling factor and unrolling strategy now come as separate assignments.
   - The register list fields use a sub‐rule (reg_list).
----------------------------------------------------------------------------*/
field:
      UNROLLING_FACTOR EQUALS NUMBER {
         printf("DICE METADATA PARSER: UNROLLING_FACTOR = %d\n", $3); fflush(stdout);
         dicemeta_parser->set_unrolling_factor($3);
      }
    | UNROLLING_STRATEGY EQUALS NUMBER {
         printf("DICE METADATA PARSER: UNROLLING_STRATEGY = %d\n", $3); fflush(stdout);
         dicemeta_parser->set_unrolling_strategy($3);
      }
    | LAT EQUALS NUMBER {
        printf("DICE METADATA PARSER: LAT = %d\n", $3); fflush(stdout);
        dicemeta_parser->set_latency($3);
      }
    | IN_REGS EQUALS reg_list {
         printf("DICE METADATA PARSER: IN_REGS\n"); fflush(stdout);
         //current_metadata->in_regs = *$3; delete $3;
      }
    | OUT_REGS EQUALS reg_list {
        printf("DICE METADATA PARSER: OUT_REGS\n"); fflush(stdout);
         //current_metadata->out_regs = *$3; delete $3;
      }
    | LD_DEST_REGS EQUALS reg_list {
        printf("DICE METADATA PARSER: LD_DEST_REGS\n"); fflush(stdout);
         //current_metadata->load_destination_regs = *$3; delete $3;
      }
    | STORE EQUALS NUMBER {
         printf("DICE METADATA PARSER: STORE = %d\n", $3); fflush(stdout);
         dicemeta_parser->set_num_store($3);
      }
    | BRANCH EQUALS NUMBER {
            printf("DICE METADATA PARSER: BRANCH = %d\n", $3); fflush(stdout);
         dicemeta_parser->set_branch($3);
      }
    | BRANCH_UNI EQUALS NUMBER {
         printf("DICE METADATA PARSER: BRANCH_UNI = %d\n", $3); fflush(stdout);
         dicemeta_parser->set_branch_uni($3);
      }
    | BRANCH_PRED EQUALS REGOPERAND {
        printf("DICE METADATA PARSER: BRANCH_PRED\n"); fflush(stdout);
         //current_metadata->branch_pred = operand_info($3);
         //free($3);
      }
    | BRANCH_TARGET EQUALS NUMBER {
            printf("DICE METADATA PARSER: BRANCH_TARGET = %d\n", $3); fflush(stdout);
         dicemeta_parser->set_branch_target_dbb($3);
      }
    | BRANCH_RECVPC EQUALS NUMBER {
            printf("DICE METADATA PARSER: BRANCH_RECVPC = %d\n", $3); fflush(stdout);
         dicemeta_parser->set_reconvergence_dbb($3);
      }
    | RET {
            printf("DICE METADATA PARSER: RET\n"); fflush(stdout);
         dicemeta_parser->set_is_ret();
      }
    ;

/*----------------------------------------------------------------------------
  A register list is enclosed in parentheses.
----------------------------------------------------------------------------*/
reg_list:
      LEFT_PAREN reg_operands RIGHT_PAREN
    ;

/*----------------------------------------------------------------------------
  A register operand list consists of one or more register tokens separated by commas.
  We build a std::list<operand_info> on the heap. (Free the temporary strings later.)
----------------------------------------------------------------------------*/
reg_operands:
      operands {printf("DICE METADATA PARSER: reg_operands: %s\n", $1); fflush(stdout);} 
    | reg_operands COMMA operands {printf("DICE METADATA PARSER: reg_operands: %s\n", $3); fflush(stdout);}
    ;

operands:
      REGOPERAND {$$=$1;}
    | SPECIAL_REGISTER DIMENSION_MODIFIER {
        $$=special_reg_names[$1]; 
        printf("DICE METADATA PARSER: special_operands:%s\n",special_reg_names[$1]);
        fflush(stdout);
    }
    ;

%%

/* End of dicemeta.y */
