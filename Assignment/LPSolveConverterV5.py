import numpy as np


def generate_arcs(Teacher_lst, Node_lst, Day_lst, Classes_lst):
    idle_arcs = np.empty((Teacher_lst, Node_lst-1, Day_lst), dtype='U20') # a11 => string for 11 chrs
    dayoff_arcs = np.empty((Teacher_lst, 1, Day_lst), dtype='U20')
    for teacher in range(Teacher_lst):
        for day in range(Day_lst):
            # b notation -> b_TA, T = Teacher, A = Day
            # Note that the number of day off arcs is not # Days - 1, since the number of day nodes is actually # Day + 1
            # The number of acrs is # nodes - 1, thus, # Days = # Dayoff arcs.
            dayoff_arcs[teacher, 0, day] = "b_{}_{}".format(teacher + 1, day + 1)
            for period in range(Node_lst-1):
                # w notation -> w_TAC, T = Teacher, A = Day, C = From Period
                idle_arcs[teacher, period, day] = "w_{}_{}_{}".format(teacher + 1, day + 1, period + 1)

    interior_arcs = np.empty((Teacher_lst, (Node_lst*2-3)*Classes_lst, Day_lst), dtype='U20')
    for teacher in range(Teacher_lst):
        for day in range(Day_lst):
            for clas in range(Classes_lst):
                for i in range(Node_lst-1):
                    # x notation -> x_TABCD, T = Teacher, A = Day, B = Class, C = From Period, D = To Period
                    interior_arcs[teacher, i + (Node_lst*2-3)*clas, day] = 'x_{}_{}_{}_{}_{}'.format(teacher + 1, day + 1, clas + 1,
                                                                                                            i + 1, i + 2)
                    # print 'interior \t\t', interior_arcs[teacher][i][day]
                for k in range(Node_lst-2):
                    interior_arcs[teacher, k + Node_lst-1 + (Node_lst*2-3)*clas, day] = 'x_{}_{}_{}_{}_{}'.format(teacher + 1, day + 1, clas + 1, k + 1, k + 3)
                    # print 'exterior \t\t', interior_arcs[teacher][k+(len(Node_lst)-1)][day]

    # Need to distinguish between initializing arcs (e.g. source to inflow node), entering arcs (e.g. inflow node to period node), leaving arcs (e.g. period node to following day "source")
    transition_arcs = np.empty((Teacher_lst, (2*Node_lst-1), Day_lst), dtype='U20')
    for teacher in range(Teacher_lst):
        for day in range(Day_lst):
            for i in range(2*Node_lst-1):
                if i == 0:
                    # i notation -> i_TA, T = Teacher, A = Day (Initializing Arc)
                    transition_arcs[teacher, i, day] = "i_{}_{}".format(teacher + 1, day + 1)
                elif i < Node_lst:
                    # e notation -> s_TAD, T = Teacher, A = Day, D = To Period (Entering Arc)
                    transition_arcs[teacher, i, day] = "e_{}_{}_{}".format(teacher + 1, day + 1, i)
                else:
                    # l notation -> e_TAC, T = Teacher, A = Day, C = From Period (Leaving Arc)
                    transition_arcs[teacher][i][day] = "l_{}_{}_{}".format(teacher + 1, day + 1, i - Node_lst + 2)

    unsatisfied_var = np.empty((Teacher_lst, Classes_lst), dtype='U20')
    for teacher in range(Teacher_lst):
        for clas in range(Classes_lst):
            unsatisfied_var[teacher, clas] = 'g_{}_{}'.format(teacher + 1, clas + 1)
    return idle_arcs, dayoff_arcs, interior_arcs, transition_arcs, unsatisfied_var


def objective_function(Teacher_lst, unsatisfied, idle, transition, delta, omega, gamma):
    objective = ""
    for teacher in range(Teacher_lst):
        obj_teach = ''
        for unsatisfied_var in unsatisfied[teacher].flatten():
            obj_teach += "{}*{} + ".format(delta, unsatisfied_var)
        for idle_arc in idle[teacher].flatten():
            obj_teach += "{}*{} + ".format(omega, idle_arc)
        for initializer_arc in transition[teacher, 0].flatten():
            obj_teach += "{}*{} + ".format(gamma, initializer_arc)
        objective += obj_teach
    return objective[:-3]


def generate_constraint_1(transition_arcs, idle_arcs, interior_arcs, dayoff_arcs, Node_lst, class_lst, day_lst):
    neutral_eqs = []
    for i in range(len(day_lst)):
        for j in range(len(Node_lst)):
            if j == (len(Node_lst)-1):
                sum_node = "-1*{}".format(transition_arcs[-1][i])
                sum_node += " + {}".format(idle_arcs[j - 1][i])
            else:
                if j == 0:
                    sum_node = transition_arcs[1 + j][i]
                    # print(idle_arcs[j][i])
                    sum_node += " - {}".format(idle_arcs[j][i])
                else:
                    sum_node = "{} - {} - {} + {}".format(transition_arcs[1 + j][i], transition_arcs[len(Node_lst) - 1 + j, i],
                                                          idle_arcs[j][i], idle_arcs[j - 1][i])
            for k in range(len(class_lst)):
                index = k*(2*len(Node_lst)-3) + j
                if j == 0:
                    sum_node += " - {} - {}".format(interior_arcs[index][i], interior_arcs[(len(Node_lst) - 1) + index][i])
                    if k == (len(class_lst)-1):
                        sum_node += " = 0;"
                        neutral_eqs.append(sum_node)
                else:
                    index_old = k * (2 * len(Node_lst) - 3) + (j - 1)
                    if j < ((len(Node_lst)-1)-1):
                        if j < 2:
                            index_old_old = k*(2*len(Node_lst)-3)
                            sum_node += " + {} - {} - {}".format(interior_arcs[index_old_old][i], interior_arcs[index][i],
                                                                 interior_arcs[(len(Node_lst) - 1) + index][i])
                            if k == (len(class_lst) - 1):
                                sum_node += " = 0;"
                                neutral_eqs.append(sum_node)
                        else:
                            sum_node += " + {} + {} - {} - {}".format(interior_arcs[index_old][i],
                                                                      interior_arcs[(len(Node_lst)-1)+index_old-1][i],
                                                                      interior_arcs[index][i], interior_arcs[(len(Node_lst)-1)+index][i])
                            if k == (len(class_lst) - 1):
                                sum_node += " = 0;"
                                neutral_eqs.append(sum_node)
                    elif j == 1:
                        sum_node += " + {} - {}".format(interior_arcs[index_old][i], interior_arcs[index][i])
                        if k == (len(class_lst) - 1):
                            sum_node += " = 0;"
                            neutral_eqs.append(sum_node)
                    elif j == ((len(Node_lst)-1)-1):
                        sum_node += " + {} + {} - {}".format(interior_arcs[index_old][i],
                                                             interior_arcs[(len(Node_lst)-1)+index_old-1][i], interior_arcs[index][i])
                        if k == (len(class_lst) - 1):
                            sum_node += " = 0;"
                            neutral_eqs.append(sum_node)
                    else:
                        index_old_old = k * (2 * len(Node_lst) - 3) + j - 2
                        sum_node += " + {} + {}".format(interior_arcs[index_old][i], interior_arcs[(len(Node_lst) - 1) + index_old_old][i])
                        if k == (len(class_lst) - 1):
                            sum_node += " = 0;"
                            neutral_eqs.append(sum_node)

    day_start_nodes = []
    dayoff_nodes = []
    for i in range(len(day_lst)):
        inflow = "{}".format(dayoff_arcs[0][i])
        if i == 0:
            outflow = "-1*{} - {} = -1;".format(dayoff_arcs[0][0], transition_arcs[0][0])
            dayoff_nodes.append(outflow)
            start = "{}".format(transition_arcs[0][i])
            for j in range(len(Node_lst)-1):
                if i == (len(day_lst) - 1):
                    inflow += " + {}".format(transition_arcs[-1 * (j + 1)][i])
                    if j == (len(Node_lst)-2):
                        inflow += " = 1;"
                        dayoff_nodes.append(inflow)
                else:
                    inflow += " + {}".format(transition_arcs[-1 * (j + 1)][i])
                    start += " - {}".format(transition_arcs[j+1][i])
                    if j == (len(Node_lst)-2):
                        inflow += " - {} - {} = 0;".format(transition_arcs[0][i + 1], dayoff_arcs[0][i + 1])
                        start += " = 0;"
                        dayoff_nodes.append(inflow)
                        day_start_nodes.append(start)
        else:
            start = "{}".format(transition_arcs[0][i])
            for j in range(len(Node_lst)-1):
                if i == (len(day_lst) - 1):
                    inflow += " + {}".format(transition_arcs[-1 * (j + 1)][i])
                    start += " - {}".format(transition_arcs[j + 1][i])
                    if j == (len(Node_lst)-2):
                        inflow += " = 1;"
                        start += " = 0;"
                        dayoff_nodes.append(inflow)
                        day_start_nodes.append(start)
                else:
                    inflow += " + {}".format(transition_arcs[-1 * (j + 1)][i])
                    start += " - {}".format(transition_arcs[j + 1][i])
                    if j == (len(Node_lst)-2):
                        inflow += " - {} - {} = 0;".format(transition_arcs[0][i + 1], dayoff_arcs[0][i + 1])
                        start += " = 0;"
                        dayoff_nodes.append(inflow)
                        day_start_nodes.append(start)
    return day_start_nodes, dayoff_nodes, neutral_eqs

def generate_constraint_2(day_lst, class_lst, Node_lst, teacher_lst, interior_arcs):
    c_lst = []
    for day in range(day_lst):
        for clas in range(class_lst):
            for period in range((Node_lst-1)):
                xt_sum = ''
                if period == (Node_lst-2):
                    index = clas*(2*Node_lst - 3) + period
                    for teacher in range(teacher_lst):
                        if teacher == 0:
                            xt_sum = interior_arcs[teacher][index][day]
                            xt_sum += "+ {}".format(interior_arcs[teacher][index + Node_lst - 2][day])
                        else:
                            xt_sum += " + {} + {}".format(interior_arcs[teacher][index][day], interior_arcs[teacher][index + Node_lst - 2][day])
                            if teacher == (teacher_lst-1):
                                xt_sum += " <= 1;"
                                c_lst.append(xt_sum)
                else:
                    if period == 0:
                        index = clas*(2*Node_lst - 3) + period
                        for teacher in range(teacher_lst):
                            if teacher == 0:
                                xt_sum = interior_arcs[teacher][index][day]
                                xt_sum += " + {}".format(interior_arcs[teacher][index + Node_lst - 1][day])
                            else:
                                xt_sum += " + {} + {}".format(interior_arcs[teacher][index][day], interior_arcs[teacher][index + Node_lst - 1][day])
                                if teacher == (teacher_lst-1):
                                    xt_sum += " <= 1;"
                                    c_lst.append(xt_sum)
                    else:
                        index = clas*(2*Node_lst - 3) + period
                        for teacher in range(teacher_lst):
                            if teacher == 0:
                                xt_sum = interior_arcs[teacher][index][day]
                                xt_sum += " + {} + {}".format(interior_arcs[teacher][index + Node_lst - 1][day], interior_arcs[teacher][index + Node_lst - 2][day])
                            else:
                                xt_sum += " + {} + {} + {}".format(interior_arcs[teacher][index][day], interior_arcs[teacher][index + Node_lst - 1][day], interior_arcs[teacher][index + Node_lst - 2][day])
                                if teacher == (teacher_lst-1):
                                    xt_sum += " <= 1;"
                                    c_lst.append(xt_sum)
    return c_lst


# Input_Duration =  [[teacher1[class1 class2 class3]]
#                    teacher2[class1 class2 class3]]]
def generate_constraint_34(day_lst, class_lst, node_lst, interior_arcs, duration, daily_duration):

    def determine_sta(string):
        return int(string[-1]) - int(string[-3])

    weighted_day_eqs = []
    weighted_day_sums = []
    c_sum = ""
    for i in range(day_lst):
        for j in range(class_lst):
            for k in range(2*node_lst-3):
                if k == 0:
                    weight = determine_sta(interior_arcs[j*(2*node_lst-3)][i])
                    c_sum = '{}*{}'.format(weight, interior_arcs[j*(2*node_lst-3)][i])
                else:
                    weight = determine_sta(interior_arcs[(j*(2*node_lst-3) + k)][i])
                    c_sum += ' + {}*{}'.format(weight, interior_arcs[(j*(2*node_lst-3) + k)][i])
                    if k == (2*node_lst - 4):
                        d_sum = '{} <= {};'.format(c_sum, daily_duration[j])
                        weighted_day_eqs.append(c_sum)
                        weighted_day_sums.append(d_sum)

    total_sum_lst = []
    cd_sum = ''
    for i in range(class_lst):
        for j in range(day_lst):
            if j == 0:
                cd_sum = '{}'.format(weighted_day_eqs[i])
            else:
                cd_sum += " + {}".format(weighted_day_eqs[j*class_lst + i])
        cd_sum += " = {};".format(duration[i])
        total_sum_lst.append(cd_sum)

    return total_sum_lst, weighted_day_sums


def generate_constraint_5(teacher_lst, day_lst, class_lst, Node_lst, interior_arcs):
    x_lst = []
    for teacher in range(teacher_lst):
        for day in range(day_lst):
            for clas in range(class_lst):
                for period in range(Node_lst-2):
                    index = period + clas*(2*Node_lst - 3)
                    x_sum = "{} + {} <= {};".format(interior_arcs[teacher][index][day], interior_arcs[teacher][index+1][day], 1)
                    x_lst.append(x_sum)
    return x_lst


def generate_constraint_6(min_double, unsatisfied_var, interior_arcs, Teacher_lst, Day_lst, Classes_lst, Node_lst):
    neg_sum_double = []
    for teacher in range(Teacher_lst):
        for day in range(Day_lst):
            for clas in range(Classes_lst):
                sum = ''
                for i in range(Node_lst - 2):
                    sum += " - {}".format(interior_arcs[teacher, clas * (2 * Node_lst - 3) + i + Node_lst - 1, day])
                neg_sum_double.append(sum)

    constr_lst = []
    for teacher in range(Teacher_lst):
        for clas in range(Classes_lst):
            constr = '{}{} <= {};'.format(min_double, neg_sum_double[3*teacher + clas], unsatisfied_var[teacher, clas])
            constr_lst.append(constr)
    return constr_lst


def generate_constraint_7(teacher_lst, day_lst, transition, min_working_days):
    constr_lst = []
    for teacher in range(teacher_lst):
        constr = ''
        for day in range(day_lst - 1):
            constr += "{} + ".format(transition[teacher, 0, day])
        else:
            constr += "{} >= {};".format(transition[teacher, 0, day_lst - 1], min_working_days)
        constr_lst.append(constr)
    return constr_lst


def generate_constraint_8(transition_arcs, idle_arcs, interior_arcs, dayoff_arcs):
    # Organise by day (each line has all day vars)
    str_lst = []
    for i in range(dayoff_arcs.shape[1]):
        # for interior arcs
        sum_int = "bin"
        for j in range(interior_arcs.shape[0] - 1):
            sum_int += " {},".format(interior_arcs[j][i])
        else:
            sum_int += "{};".format(interior_arcs[interior_arcs.shape[0] - 1, i])

        # for transition arcs
        sum_trans = "bin"
        for j in range(transition_arcs.shape[0] - 1):
            sum_trans += " {},".format(transition_arcs[j][i])
        else:
            sum_trans += "{};".format(transition_arcs[transition_arcs.shape[0] - 1, i])

        # for idle arcs
        sum_idle = "bin"
        for j in range(idle_arcs.shape[0] - 1):
            sum_idle += " {},".format(idle_arcs[j][i])
        else:
            sum_idle += "{};".format(idle_arcs[idle_arcs.shape[0] - 1, i])

        # for idle arcs
        sum_do = "bin "
        for j in range(dayoff_arcs.shape[0] - 1):
            sum_do += " {},".format(dayoff_arcs[j][i])
        else:
            sum_do += "{};".format(dayoff_arcs[dayoff_arcs.shape[0] - 1, i])

        str = "{}\n{}\n{}\n{}".format(sum_int, sum_trans, sum_idle, sum_do)
        str_lst.append(str)
    return str_lst


number_teachers = 2
teachers = np.arange(1, number_teachers + 1, 1)

number_periods = 4
periods = np.arange(1, number_periods + 1, 1)

number_days = 4
days = np.arange(1, number_days + 1, 1)

number_classes = 2
classes = np.arange(1, number_classes + 1, 1)

min_num_double_lessons = 2

max_hrs = np.ones((number_teachers, number_classes))*10
max_hrs[0][0] = 6
max_hrs[1][1] = 6
max_hrs_per_day = np.ones((number_teachers, number_classes))*number_periods

min_working_days = 1

delta = 1
omega = 3
gamma = 9

idle, dayoff, interior, transition, unsatisfied = generate_arcs(number_teachers, (number_periods + 1), number_days, number_classes)
# print(idle)
# print(dayoff)
# print(interior)
# print(transition)
# print(unsatisfied)

print('\nTotal number of variables: ', len(idle.flatten())
                                     + len(dayoff.flatten())
                                     + len(interior.flatten())
                                     + len(transition.flatten())
                                     + len(unsatisfied.flatten()))

print_objective = True
print_constraint_1 = True
print_constraint_2 = True
print_constraint_3 = True
print_constraint_4 = True
print_constraint_5 = True
print_constraint_6 = True
print_constraint_7 = True
print_constraint_8 = True

link_lst = []
b_lst = []
x_lst = []
nodes = np.hstack((periods,number_periods+1))
for i in range(number_teachers):
    links, bs, xs = generate_constraint_1(transition[i], idle[i], interior[i], dayoff[i], nodes, classes, days)
    link_lst.append(links)
    b_lst.append(bs)
    x_lst.append(xs)

objective = objective_function(number_teachers, unsatisfied, idle, transition, delta, omega, gamma)

total_duration_constraint = []
daily_duration_constraints = []
for i in range(number_teachers):
    tot_duration, daily_duration = generate_constraint_34(number_days, number_classes, (number_periods+1), interior[i], max_hrs[i], max_hrs_per_day[i])
    total_duration_constraint.append(tot_duration)
    daily_duration_constraints.append(daily_duration)

constraint_2 = generate_constraint_2(number_days, number_classes, (number_periods+1), number_teachers, interior)

constraint_5 = generate_constraint_5(number_teachers, number_days, number_classes, (number_periods+1), interior)

constraint_6 = generate_constraint_6(min_num_double_lessons, unsatisfied,
                                     interior, number_teachers, number_days, number_classes, (number_periods+1))

constraint_7 = generate_constraint_7(number_teachers, number_days, transition, min_working_days)

if print_objective:
    print("\n/*--------------------- OBJECTIVE FUNCTION ---------------------*/")
    print("min: {};".format(objective))

if print_constraint_1:
    print("\n/*--------------------- CONSTRAINT 1 - Node inflow and outflow for a given teacher ---------------------*/")
    for teacher in range(number_teachers):
        print("\n/* Teacher {} */".format(teacher + 1))
        print("/* Day off Nodes: */")
        for i in b_lst[teacher]:
            print(i)
        print("\n/* Day Start Nodes: */")
        for i in link_lst[teacher]:
            print(i)
        print("\n/* Internal (Period) Nodes: */")
        for i in x_lst[teacher]:
            print(i)

if print_constraint_2:
    print("\n/*--------------------- CONSTRAINT 2 - One Teacher per Arc ---------------------*/")
    # print len(constraint_2), (2*number_periods-3)*number_classes*number_days
    for i in constraint_2:
        print(i)

if print_constraint_3:
    print("\n/*--------------------- CONSTRAINT 3 - Total Hours Constraint ---------------------*/")
    for teacher in range(number_teachers):
        print("\n/* Teacher {} */".format(teacher + 1))
        for i in total_duration_constraint[teacher]:
            print(i)

if print_constraint_4:
    print("\n/*--------------------- CONSTRAINT 4 - Duration Constraints, per day ---------------------*/")
    for teacher in range(number_teachers):
        print("\n/* Teacher {} */".format(teacher + 1))
        for i in daily_duration_constraints[teacher]:
            print(i)

if print_constraint_5:
    print("\n/*--------------------- CONSTRAINT 5 - No Consecutive Single Arcs ---------------------*/")
    for i in constraint_5:
        print(i)

if print_constraint_6:
    print("\n/*--------------------- CONSTRAINT 6 - Non-negative Unsatisfied Double Lessons ---------------------*/")
    for i in constraint_6:
        print(i)

if print_constraint_7:
    print("\n/*--------------------- CONSTRAINT 7 - Minimum Working Days ---------------------*/")
    for i in constraint_7:
        print(i)

if print_constraint_8:
    print("\n/*--------------------- CONSTRAINT 8 - Binary Variable Constraints ---------------------*/")
    for teacher in range(number_teachers):
        print("\n\n/* Teacher {} */".format(teacher + 1))
        str_lst = generate_constraint_8(transition[teacher], idle[teacher], interior[teacher], dayoff[teacher])
        for i in range(len(str_lst)):
            print("\n/* Binary Constraints for day {}*/".format(i+1))
            print(str_lst[i])