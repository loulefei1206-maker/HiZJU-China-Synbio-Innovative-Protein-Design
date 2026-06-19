import csv

log_file_path = "esmfold_45672.out"  # 请确保你的日志文件名和这里一致
output_csv_path = "avGFP_stability_ranked.csv"

parsed_data = []

print("开始逐行扫描日志...")
with open(log_file_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        # 只要这一行包含我们想要的输出关键字
        if "Predicted structure for" in line and "pLDDT" in line:
            try:
                # 1. 提取突变体名称
                # 根据 "Predicted structure for " 切分，拿后面那一半
                parts = line.split("Predicted structure for ")
                variant_part = parts[1].split(" with length")[0].strip()

                # 2. 提取 pLDDT 分数
                plddt_part = line.split("pLDDT ")[1].split(",")[0].strip()

                # 3. 提取 pTM 分数
                ptm_part = line.split("pTM ")[1].split(" ")[0].strip()
                ptm_part = ptm_part.replace(",", "")  # 去掉可能残存的逗号

                # 4. 提取时间
                time_part = line.split(" in ")[1].split("s.")[0].strip()

                # 组装数据
                parsed_data.append(
                    {
                        "Variant": variant_part,
                        "pLDDT": float(plddt_part),
                        "pTM": float(ptm_part),
                        "Time_seconds": float(time_part),
                    }
                )
            except Exception as e:
                print(
                    f"⚠️ 第 {line_num} 行解析时发生轻微错位，已跳过。错误原因: {e}"
                )

# 按 pLDDT 从高到低排序
parsed_data.sort(key=lambda x: x["pLDDT"], reverse=True)

# 写入 CSV
if parsed_data:
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Variant", "pLDDT", "pTM", "Time_seconds"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in parsed_data:
            writer.writerow(row)
    print(f"✅ 成功！抓取到 {len(parsed_data)} 条数据，已保存至 {output_csv_path}")
else:
    print(
        "❌ 依旧没有读取到数据！请检查当前目录下是否存在 'esmfold_run.log'，或者该文件内部是否为空。"
    )
