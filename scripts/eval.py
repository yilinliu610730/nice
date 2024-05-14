import sys

sys.path.append(".")

from nice.eval import run_eval, compute_cider, merge_gt_pred

def main():

    test_gt = 'data/nice-val-5k.csv'
    test_data = "data/nice-val-5k"
    out_file = "pred.csv"

    # run_eval(img_dir=test_data)
    cider_score = compute_cider(test_gt, out_file)
    print(cider_score)

    merge_gt_pred(test_gt, out_file)

if __name__ == '__main__':
    main()