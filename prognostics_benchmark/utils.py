def parse_rtf_id(rtf_id):
    rtf_id_split = rtf_id.split('_')
    model_id = rtf_id_split[0]
    equi_id = rtf_id_split[1]
    run_idx = rtf_id_split[2]
    if str.isdigit(model_id):
        model_id = int(model_id)
    if str.isdigit(equi_id):
        equi_id = int(equi_id)
    return model_id, equi_id, int(run_idx)