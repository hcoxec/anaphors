function(command_line_args={}) {
    name: 'anaphors',
    wandb_name: 'anaphoric_structure_no_cost',
    project_name: 'anaphoric_structure_no_cost',
    local _default_args = {
        #Defaults that can be overwritten by command line args
        exp_id: 1000,
        att: 2, #seed
        params: 250,
        logdir: 'anaphor_experiments', #checkpoint directory
        wkdir: 'anaphor_experiments', #logged results directory
        eval_section: 'test_mcd',  # gen_samples, gen, test
        sender_entropy: 0.5,
        decay_coeff: 0.0,
        dropout: 0.0,
        mode: null,
        task: 'train',
        max_epochs: 800,
    },
    args: _default_args + command_line_args,

    #DATA
    dataset: 'redundant_predicates',
    data_scaling: 20,
    lang_params: [40,40,15,15], #if generating new data: 1+2: no.nouns/verbs to generate; 3+4: no.nouns/verbs to use
    setting: 'redlarge',
    gram_fn: 'redlarge',
    red: true, 
    probs: 'uniform',
    n_roles: 5,
    n_atoms: 32, #total across all roles

    #TRAINING
    seed: $.args.att,
    run_id: ['hidden', $.args.params, 'd', $.args.dropout, 'l', $.args.decay_coeff],
    epochs: 3000,
    batch_size: 5000,
    learning_rate: 0.001,
    sender_entropy: $.args.sender_entropy,
    decay_coeff: $.args.decay_coeff,
    sender_dropout: $.args.dropout,
    receiver_dropout: $.args.dropout,
    callbacks:['saver', 'analysis'], #'console', 'progress', 'wandb'
    check_every: 20, #checkpoint frequency
    validation_freq: 100, #validation frequency
    update_freq: 1,
    use_wandb: false,

    notebook: false, #tries to do some stuff to make things run well in nb

    #MODEL
    rnn_cell: 'gru',
    hidden_size: $.args.params,
    embedding_size: 32,
    signal_len: 10,
    signal_alphabet_size: 26,
    length_cost: 0.0, #to add length cost, change to 0.15 (as used in the paper)
}