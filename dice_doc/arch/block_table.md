# CTA Table

CTA table is to hold the status of current active blocks in a cgra core. It's a table with max_num_cta entries. 

Each entry holds informations such as **cta_id.xyz**, **grid_size.xyz**, **cta_size.xyz** ,**threads_per_cta** , **kernel_id**(for future concurrent kernel running). 

The **hw_cta_id** is the index in the CTA table.

For each block, the hardware thread id is calculated by **hw_cta_id * threads_per_cta**.

## Issue a block to cgra core

If there's an empty entry in CTA table (active number of CTAs < number of table entries), then the cgra core can accept a new CTA running. 

- CTA initial infomation (**cta_id**, start **pc**, start **active_mask**) is sent from top-level to the table if current core can accept a new CTA.
- Once the new CTA infomation is received, the CTA table finds an empty slot in table and fill the CTA in.
- Set the initial corresponding PDOM stack top for this CTA.

## Pop a finished CTA

If a CTA finishes (which means the stack top active mask is all 0), then it's flushed out in CTA table and send a signal back to top level through interconnect.