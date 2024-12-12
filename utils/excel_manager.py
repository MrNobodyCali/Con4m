import os
import xlwt
import xlrd


class ExcelManager:
    def __init__(self, metr_save_path, filename):
        self.metr_save_path = metr_save_path
        self.filename = filename
        self.workbook = xlwt.Workbook(encoding='utf-8')
        self.worksheet = self.workbook.add_sheet('result')
        self.workbook_summary = xlwt.Workbook(encoding='utf-8')
        self.worksheet_loss_summary = self.workbook_summary.add_sheet('loss_summary')
        self.worksheet_f1_summary = self.workbook_summary.add_sheet('f1_summary')
        self.worksheet_last_summary = self.workbook_summary.add_sheet('last_summary')
        self.row_cursor = 0

    def res2excel(self, res, tar_pat_name, tag=''):
        self.worksheet.write(self.row_cursor, 0, label=self.filename)
        self.worksheet.write(self.row_cursor, 1, label=tar_pat_name)
        self.worksheet.write(self.row_cursor, 2, label=tag)
        self.row_cursor += 1

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        metr_num = len(metrics)
        for i, metr in enumerate(metrics):
            self.worksheet.write(self.row_cursor, i, label=metr)
        self.row_cursor += 1

        lines = res.split('\n')
        values = []
        for line in lines:
            if ':' not in line:
                continue
            line = line.strip()
            k, v = line.split(':')
            if k in metrics:
                values.append(v)
        for idx, v in enumerate(values):
            self.worksheet.write(self.row_cursor + idx // metr_num, idx % metr_num, v)

        self.row_cursor += 2

    def excel_save(self, step):
        path = os.path.join(self.metr_save_path, f'{self.filename}_{step}.xls')
        self.workbook.save(path)
        print(f'Excel has saved to {path}')

    def summary_results(self):
        prefix = os.path.split(self.metr_save_path)[0]
        row_num = 0
        for exp_id in range(1, 13):
            save_path = os.path.join(prefix, f'exp{exp_id}')
            read_file = None
            if not os.path.exists(save_path):
                continue
            for file in os.listdir(save_path):
                if file.split('.')[-1] == 'xls':
                    read_file = file
                    break
            if read_file is not None:
                workbook = xlrd.open_workbook(os.path.join(save_path, read_file))
                worksheet = workbook.sheet_by_name('result')
                # Best loss model
                row_value = worksheet.row_values(2)
                for col in range(len(row_value)):
                    self.worksheet_loss_summary.write(row_num, col, row_value[col])

                if worksheet.nrows > 4:
                    # Best F1 model
                    row_value = worksheet.row_values(6)
                    for col in range(len(row_value)):
                        self.worksheet_f1_summary.write(row_num, col, row_value[col])
                    # Best last model
                    row_value = worksheet.row_values(10)
                    for col in range(len(row_value)):
                        self.worksheet_last_summary.write(row_num, col, row_value[col])
            else:
                print(f'Warning: file of exp{exp_id} missing.')
            row_num += 1

        summary_save_path = os.path.join(prefix, f'{self.filename}_summary.xls')
        self.workbook_summary.save(summary_save_path)
        print(f'Excel has saved to {summary_save_path}')
