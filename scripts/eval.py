import sys

sys.path.append(".")

from nice.eval import run_eval, compute_cider, merge_gt_pred

def main():

    test_data = "data/nice-val-5k"
    test_gt = 'data/nice-val-5k.csv'
    ofa_pred_file = "ofa_pred.csv"
    blip_pred_file = "blip_pred.csv"

    run_eval(img_dir=test_data, model="ofa", out_file=ofa_pred_file)
    run_eval(img_dir=test_data, model="blip2", out_file=blip_pred_file)


    ofa_cider_score = compute_cider(test_gt, pred_file=ofa_pred_file)
    blip_cider_score = compute_cider(test_gt, pred_file=blip_pred_file)
    
    print(f"ofa_cider_score: {ofa_cider_score}")
    print(f"blip_cider_score: {blip_cider_score}")

    merge_gt_pred(test_gt, ofa_pred_file, out_file="ofa_merged.csv")
    merge_gt_pred(test_gt, blip_pred_file, out_file="blip_merged.csv")

if __name__ == '__main__':
    main()