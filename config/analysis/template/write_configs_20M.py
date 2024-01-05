import numpy as np

header_str = "wandb_project = 'SASP-25M-analysis'\ndataset = 'shakespeare_char'\neval_interval = 200\neval_iters = 100\nlog_interval = 10\nalways_save_checkpoint = False\nwandb_log = True\ngradient_accumulation_steps = 1\nbatch_size = 128\nblock_size = 256\nn_layer = 8\nn_head = 8\nlearning_rate = 0.2e-3\nmax_iters = 45000\nlr_decay_iters = 45000\nmin_lr = 0.2e-4\nbeta2 = 0.99\nwarmup_iters = 2000\n"

wandb_run_configs = [ ['SASP-dr_0.1'   , 'SASP' , False, 552, 0.1],
    ['SASP-dr_0.2'   , 'SASP' , False, 552, 0.2],
    ['SASP-dr_0.025' , 'SASP' , False, 552, 0.025],
    ['SASPV-dr_0.1'  , 'SASP' , True , 520, 0.1],
    ['SASPV-dr_0.2'  , 'SASP' , True , 520, 0.2],
    ['SASPV-dr_0.025', 'SASP' , True , 520, 0.025],
    ['PreLN-dr_0.1'  , 'PreLN', True , 520, 0.1],
    ['PreLN-dr_0.2'  , 'PreLN', True , 520, 0.2],
    ['PreLN-dr_0.025', 'PreLN', True , 520, 0.025] ]

dropout_values = [0.25, 0.1, 0.2]

count = 0
write_str = []
for run_config in wandb_run_configs:
    wandb_run_name = run_config[0] 
    trans_block    = run_config[1] 
    use_v          = run_config[2] 
    n_embd         = run_config[3]  
    dropout        = run_config[4]

    write_str.append('')
    write_str[count] = "transformer_block_type = '" + trans_block + "'\nwandb_run_name = '"  
    write_str[count] = write_str[count] + wandb_run_name + "'\nout_dir = 'analysis/size_25M/" + wandb_run_name + "'\n" 
    write_str[count] = write_str[count] + header_str + "dropout = " + str(dropout) + "\nuse_v = " + str( use_v ) 
    write_str[count] = write_str[count] + "\nn_embd = " + str(n_embd)

    filename = "tmp/" + wandb_run_name + ".py"
    with open( "tmp/" + wandb_run_name + ".py", 'w') as f:
        f.write(write_str[count])
        print(write_str[count])
        print('#################################################')
        print('count = ', count)
        print('#################################################')
        f.close()
    count = count + 1
