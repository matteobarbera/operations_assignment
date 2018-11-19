import numpy as np
import pandas as pd


def extract_results_robust(csv_filename):
    data = np.array(pd.read_csv(csv_filename, delimiter = ";", usecols = ['Variables', "result"]))

    result = data[np.argwhere(data[:,-1])]
    vals = np.zeros((len(result), 2), dtype = 'a20')
    for i in range(len(result)):
        vals[i][0], vals[i][1] = result[i][0][0], result[i][0][1]

    obj = vals[0, 1]
    active_nodes = vals[1:-1]

    return obj, active_nodes


def create_timetable(arcs, number_periods, number_days):
    schedule = np.zeros((number_periods, number_days), dtype='a50')

    def extract_nodes(arcs, reference, sorting_index, accept_g=False):
        output = []
        if accept_g:
            for i in range(len(arcs)):
                if str(arcs[i, 0][sorting_index]) == str(reference):
                    output.append(arcs[i][0])
        else:
            for i in range(len(arcs)):
                if arcs[i, 0][0] != 'g':
                    if str(arcs[i, 0][sorting_index]) == str(reference):
                        output.append(arcs[i][0])
        return output

    def extract_vars(variable, list, index):
        arcs = []
        for i in xrange(len(list)):
            if str(list[i][index]) == '{}'.format(variable):
                arcs.append(list[i])
        return arcs

    def determine_sta(string):
        try:
            output = int(string[-1]) - int(string[-3])
        except:
            output = 0
        return output

    for day in xrange(number_days):
        day_lst = extract_nodes(arcs, day + 1, 4)
        scanning, day_start = True, False
        scan = 0
        while scanning:
            if str(day_lst[scan][0][0]) == 'i':
                scanning, day_start = False, True
            scan += 1
        if day_start:
            x_arcs = extract_vars('x', day_lst, 0)
            w_arcs = extract_vars('w', day_lst, 0)
            if len(w_arcs) <= 0:
                no_idle = True
            else:
                no_idle = False
            if no_idle:
                for period in xrange(number_periods):
                    period_x_lst = extract_vars('{}'.format(period+1), x_arcs, 8)
                    if period == 0:
                        if len(period_x_lst) >= 1:
                            if determine_sta(period_x_lst[0]) != 2:
                                info = 'T{}C{}'.format(period_x_lst[0][2], period_x_lst[0][6])
                            else:
                                info = 'Dbl T{}C{}'.format(period_x_lst[0][2], period_x_lst[0][6])
                            for event in xrange(len(period_x_lst)-1):
                                if determine_sta(period_x_lst[event + 1]) != 2:
                                    info += ', T{}C{}'.format(period_x_lst[event + 1][2], period_x_lst[event + 1][6])
                                else:
                                    info += ', Dbl T{}C{}'.format(period_x_lst[event + 1][2],
                                                                  period_x_lst[event + 1][6])
                            schedule[period][day] = info
                        else:
                            schedule[period][day] = '-'
                    else:
                        if len(period_x_lst) >= 1:
                            if determine_sta(period_x_lst[0]) != 2:
                                info = 'T{}C{}'.format(period_x_lst[0][2], period_x_lst[0][6])
                            else:
                                info = 'Dbl T{}C{}'.format(period_x_lst[0][2], period_x_lst[0][6])
                            for event in xrange(len(period_x_lst)-1):
                                if determine_sta(period_x_lst[event + 1]) != 2:
                                    info += ', T{}C{}'.format(period_x_lst[event + 1][2], period_x_lst[event + 1][6])
                                else:
                                    info += ', Dbl T{}C{}'.format(period_x_lst[event + 1][2],
                                                                  period_x_lst[event + 1][6])
                            old_period_x_lst = extract_vars('{}'.format(period), x_arcs, 8)
                            if len(old_period_x_lst) >= 1:
                                if determine_sta(old_period_x_lst[0]) == 2:
                                    if len(info) > 1:
                                        info += ', Dbl T{}C{}'.format(old_period_x_lst[0][2], old_period_x_lst[0][6])
                                    else:
                                        info = 'Dbl T{}C{}'.format(old_period_x_lst[0][2], old_period_x_lst[0][6])
                                for event in xrange(len(old_period_x_lst) - 1):
                                    if determine_sta(old_period_x_lst[event + 1]) == 2:
                                        info += ', Dbl T{}C{}'.format(old_period_x_lst[event + 1][2],
                                                                      old_period_x_lst[event + 1][6])
                            schedule[period][day] = info
                        else:
                            old_period_x_lst = extract_vars('{}'.format(period), x_arcs, 8)
                            info = '-'
                            if len(old_period_x_lst) >= 1:
                                if determine_sta(old_period_x_lst[0]) == 2:
                                    info = 'Dbl T{}C{}'.format(old_period_x_lst[0][2], old_period_x_lst[0][6])
                                for event in xrange(len(old_period_x_lst) - 1):
                                    if determine_sta(old_period_x_lst[event + 1]) == 2:
                                        if len(info) >= 2:
                                            info += ', Dbl T{}C{}'.format(old_period_x_lst[event + 1][2],
                                                                          old_period_x_lst[event + 1][6])
                                        else:
                                            info = 'Dbl T{}C{}'.format(old_period_x_lst[event + 1][2],
                                                                       old_period_x_lst[event + 1][6])
                            schedule[period][day] = info

            else:
                for period in xrange(number_periods):
                    info = '-'
                    period_w_lst = extract_vars('{}'.format(period + 1), w_arcs, 6)
                    if len(period_w_lst) >= 1:
                        info = 'T{} Idle'.format(period_w_lst[0][2])
                        for event in xrange(len(period_w_lst)-1):
                            info += ', T{} Idle'.format(period_w_lst[event + 1][2])
                        schedule[period][day] = info
                    period_x_lst = extract_vars('{}'.format(period+1), x_arcs, 8)
                    if period == 0:
                        if len(period_x_lst) >= 1:
                            if len(info) > 1:
                                if determine_sta(period_x_lst[0]) != 2:
                                    info = ', T{}C{}'.format(period_x_lst[0][2], period_x_lst[0][6])
                                else:
                                    info = ', Dbl T{}C{}'.format(period_x_lst[0][2], period_x_lst[0][6])
                            else:
                                if determine_sta(period_x_lst[0]) != 2:
                                    info = 'T{}C{}'.format(period_x_lst[0][2], period_x_lst[0][6])
                                else:
                                    info = 'Dbl T{}C{}'.format(period_x_lst[0][2], period_x_lst[0][6])
                            for event in xrange(len(period_x_lst)-1):
                                if determine_sta(period_x_lst[event + 1]) != 2:
                                    info += ', T{}C{}'.format(period_x_lst[event + 1][2], period_x_lst[event + 1][6])
                                else:
                                    info += ', Dbl T{}C{}'.format(period_x_lst[event + 1][2],
                                                                  period_x_lst[event + 1][6])
                            schedule[period][day] = info
                        else:
                            schedule[period][day] = '-'
                    else:
                        if len(period_x_lst) >= 1:
                            if len(info) > 1:
                                if determine_sta(period_x_lst[0]) != 2:
                                    info = ', T{}C{}'.format(period_x_lst[0][2], period_x_lst[0][6])
                                else:
                                    info = ', Dbl T{}C{}'.format(period_x_lst[0][2], period_x_lst[0][6])
                            else:
                                if determine_sta(period_x_lst[0]) != 2:
                                    info = 'T{}C{}'.format(period_x_lst[0][2], period_x_lst[0][6])
                                else:
                                    info = 'Dbl T{}C{}'.format(period_x_lst[0][2], period_x_lst[0][6])
                            for event in xrange(len(period_x_lst)-1):
                                if determine_sta(period_x_lst[event + 1]) != 2:
                                    info += ', T{}C{}'.format(period_x_lst[event + 1][2], period_x_lst[event + 1][6])
                                else:
                                    info += ', Dbl T{}C{}'.format(period_x_lst[event + 1][2],
                                                                  period_x_lst[event + 1][6])
                            old_period_x_lst = extract_vars('{}'.format(period), x_arcs, 8)
                            if len(old_period_x_lst) >= 1:
                                if determine_sta(old_period_x_lst[0]) == 2:
                                    if len(info) > 1:
                                        info += ', Dbl T{}C{}'.format(old_period_x_lst[0][2], old_period_x_lst[0][6])
                                    else:
                                        info = 'Dbl T{}C{}'.format(old_period_x_lst[0][2], old_period_x_lst[0][6])
                                for event in xrange(len(old_period_x_lst) - 1):
                                    if determine_sta(old_period_x_lst[event + 1]) == 2:
                                        info += ', Dbl T{}C{}'.format(old_period_x_lst[event + 1][2],
                                                                      old_period_x_lst[event + 1][6])
                            schedule[period][day] = info
                        else:
                            old_period_x_lst = extract_vars('{}'.format(period), x_arcs, 8)
                            info = '-'
                            if len(old_period_x_lst) >= 1:
                                if determine_sta(old_period_x_lst[0]) == 2:
                                    info = 'Dbl T{}C{}'.format(old_period_x_lst[0][2], old_period_x_lst[0][6])
                                for event in xrange(len(old_period_x_lst) - 1):
                                    if determine_sta(old_period_x_lst[event + 1]) == 2:
                                        if len(info) >= 2:
                                            info += ', Dbl T{}C{}'.format(old_period_x_lst[event + 1][2],
                                                                          old_period_x_lst[event + 1][6])
                                        else:
                                            info = 'Dbl T{}C{}'.format(old_period_x_lst[event + 1][2],
                                                                       old_period_x_lst[event + 1][6])
                            schedule[period][day] = info

    period_lst = np.ones((number_periods, 1))
    period_range = np.array(np.arange(1, number_periods + 1, 1))
    for i in range(len(period_lst)):
        period_lst[i] = period_range[i]

    schedule_periods = np.hstack((period_lst, schedule))
    day_lst = np.ones((1, number_days + 1), dtype='a20')
    day_range = np.array(np.arange(1, number_days + 1, 1))

    for i in range(len(day_lst[0])):
        if i == 0:
            day_lst[0][i] = "Period"
        else:
            day_lst[0][i] = "Day {}".format(day_range[i - 1])

    tt = np.vstack((day_lst, schedule_periods))

    return tt

# After running the LPSolve Model, we may save the results as a csv file. This program uses that csv file to create
# a rudimentary timetable.
file_name = 'Own2.csv'

# Calls the extract results function which reads the csv file and retrieves all the active arcs, and the objective value
obj, active_arcs = extract_results_robust(file_name)

# Here, the number of periods and days of the model need to be inputted, these are needed to produce the timetable
periods = 5
days = 5

# Here, a timetable is created by calling the function "create_timetable". This timetable is an overall timetable and
# includes all classes for a given period and day. The duration of each lesson is also considered; in that, a lesson
# spanning periods 1 and 2 will appear in periods 1 and 2. To more easily distinguish between double and single lessons
# the term "Dbl" is prefixed to the teacher class combination. Each teacher class combination (i.e. event) is also
# specified as TiCj where Ti gives the teacher, and Cj the class. For example, teacher 1 teaching class 3 will be noted
# as T1C3. Moreover, idle arcs are also isolated by means of "Ti Idle".
timetable = create_timetable(active_arcs, periods, days)

# Here, the data is exported into a csv file such that it can be read and manipulated in excel, which gives a much more
# intuitive interface by which to work with the timetable.
data_frame = pd.DataFrame(timetable)
data_frame.to_csv('Timetable {}'.format(file_name))