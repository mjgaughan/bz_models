import csv




#how long/detailed is the body of the function
def body_len(body):
    return len(body)

def find_left_invoke(body, param):
    body_in_rows = body.split("\n")
    param_name = param.split()[-1] 
    if "*" in param_name:
        #checking if the pointer is being used? might need to get a bit more sophisticated on this and look at how pointers are invoked
        param_name = param_name[1:]
    for line in body_in_rows:
        if param_name in line and "=" in line:
            #if the param is being assigned, check w syntax location and whether the thing is standing alone
            if (line.index(param_name) < line.index("=")) and (param_name + " ") in line:
                #print("bozo")
                return True
    return False




if __name__ == "__main__":
    with open("../bz_func_declarations/temp_final_labeled_body_shuffled.csv") as fp:
        location = 0
        rows = csv.reader(fp)
        header = next(rows)
        for row in rows:
            location += 1
            datapoint = {}
            for (column_name, column_value) in zip(header,row):
                datapoint[column_name] = column_value
            #print(datapoint)
            target_param = ""
            #finding the targeted param
            for i in range(10):
                current_param_entry = datapoint["a" + str(i)]
                if current_param_entry != "ignore" and current_param_entry != "u":
                    target_param = current_param_entry
                    break
            body_length = body_len(datapoint["func_body_text"])
            print(body_length)
            find_left_invoke(datapoint["func_body_text"], target_param)
            if location == 3:
                break
