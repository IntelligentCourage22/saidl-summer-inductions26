def compute_fid(real_dir, generated_dir):
    from cleanfid import fid

    return fid.compute_fid(real_dir, generated_dir, mode="clean")
