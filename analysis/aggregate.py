rows_A = open("/home/lijunlin/lijunlin/ESCONV_ACL/analysis/results/summary.txt","r+").read().split("\n")
rows_B = open("/home/lijunlin/lijunlin/ESCONV_ACL/analysis/results/summary_relv.txt","r+").read().split("\n")
rows_B = [x.split("\t")[-1] for x in rows_B]
new_rows = [f"{x}\t{y}" for x,y in zip(rows_A, rows_B)]
with open("results/summary_all.txt", "w+") as file:
    for row in new_rows:
        file.write(row)
        file.write("\n")