import numpy as np
import pandas as pd


df = pd.read_csv("FatFixed.csv")


def check_allergy(allergy):
    result = df[~df["ingredients"].str.contains(allergy)]
    return result


# df = check_allergy(allergy, df)

id = df["id"].values
Fat = df["Fat"].values
Carbon = df["Carbon"].values
Protein = df["Protein"].values
dishtypes = df["dishTypes"].values


def id_to_matrix(arr):
    matrix = []
    for i in arr:
        row = [i, Protein[i], Carbon[i], Fat[i]]
        matrix.append(row)
    return np.array(matrix)


def sum_nutrition(x):
    return x.sum(axis=0)[1:4]


def auto_gen_meal(N, dishtypes, flag):
    if flag == True:
        e = [
            i
            for i, x in enumerate(dishtypes)
            if x == "lunch,main course,main dish,dinner"
        ]
        id = np.random.choice(e, N)
    else:
        e = [
            i
            for i, x in enumerate(dishtypes)
            if x != "lunch,main course,main dish,dinner"
        ]
        id = np.random.choice(e, N)
    return id_to_matrix(id)


def meal_eval(df1, p, ca, f):
    a, b, c = sum_nutrition(df1)
    eval = np.abs(a - f) + np.abs(b - ca) + np.abs(c - p)
    return eval


def reset_basket(dfs):
    dfs = pd.DataFrame(columns=df.columns)
    return dfs


def fill_meal(meal, protein, carbon, fat):
    df = auto_gen_meal(meal, dishtypes, True)
    loss = 15
    iter = 0
    a, b, c = 0, 0, 0
    while not (
        (fat - 20 < c < fat + 15)
        and (protein - loss < a < protein + loss)
        and (carbon - loss < b < carbon + loss)
    ):
        df = auto_gen_meal(meal, dishtypes, True)
        new_row = auto_gen_meal(1, dishtypes, False)
        df = np.r_[df, new_row]
        iter += 1
        if iter > 16:
            df = auto_gen_meal(2, dishtypes, True)
            iter = 0
        a, b, c = sum_nutrition(df)
    return df


def hill_climbing(N, protein, carbon, fat, allergy):
    check_allergy(allergy)
    optimal = fill_meal(N, protein, carbon, fat)
    print(optimal)
    iter = 0
    for i in range(0, 20):
        df1 = fill_meal(N, protein, carbon, fat)
        if meal_eval(df1, protein, carbon, fat) <= meal_eval(
            optimal, protein, carbon, fat
        ):
            optimal = df1
    p, ca, f = sum_nutrition(optimal)
    print(p, ca, f)
    return optimal


# def File(file):
#     # filename = "/Python_expert_sys3/website/Clean_food.csv"
#     df = pd.read_csv(file, header=0)
#     return df


# def preprocess(df):
#     # fat = df.weightPerServing * df["percentFat"] / 100
#     # Carbon = df.weightPerServing * df["percentCarbs"] / 100
#     # Protein = df.weightPerServing * df["percentProtein"] / 100

#     df = df[["id", "title", "weightPerServing", "dishTypes", "calories", "Fat", "Carbon", "Protein"]]
#     # df["Fat"] = fat
#     # df["Carbon"] = Carbon
#     # df["Protein"] = Protein

#     return df


# output = preprocess(
#     File("E:/DSS/Python_expert_sys3/website/FatFixed.csv")
# )


# def auto_gen_meal(N, df):
#     df_cal = df[df["dishTypes"] == "lunch,main course,main dish,dinner"].sample(
#         n=N, replace=True
#     )
#     return df_cal


# def sum_nutrition(df_cal):
#     sumProtein = np.sum(df_cal["Protein"])
#     sumCarbon = np.sum(df_cal["Carbon"])
#     sumFat = np.sum(df_cal["Fat/g"])
#     return sumProtein, sumCarbon, sumFat


# def meal_eval(df1, p, ca, f):
#     a, b, c = sum_nutrition(df1)
#     eval = np.abs(a - f) + np.abs(b - ca) + np.abs(c - p)
#     return eval


# def reset_basket(dfs):
#     dfs = pd.DataFrame(columns=output.columns)
#     return dfs


# def fill_meal(df, protein, carbon, fat):
#     iter = 0
#     a, b, c = 0, 0, 0
#     while ((a < protein) or (b < carbon) or (c < fat)):
#         df_cal = auto_gen_meal(2, df)
#         new_row = df[df["dishTypes"] != "lunch,main course,main dish,dinner"].sample(
#             n=1, replace=True
#         )
#         df_cal = df_cal.append(new_row, ignore_index=False)
#         iter += 1
#         if iter > 16:
#             reset_basket(df)
#             iter = 0
#         a, b, c = sum_nutrition(df_cal)
#     return df_cal


# def hill_climbing(protein, carbon, fat):
#     optimal = fill_meal(output, protein, carbon, fat)
#     iter = 0
#     for i in range(0, 10):
#         df1 = fill_meal(output, protein, carbon, fat)
#         if meal_eval(df1, protein, carbon, fat) <= meal_eval(
#             optimal, protein, carbon, fat
#         ):
#             optimal = df1
#         print(iter)
#         iter += 1
#     p, ca, f = sum_nutrition(optimal)
#     print(p, ca, f)
#     return optimal


def main():
    N = 3
    protein, carbon, fat = 130.0, 131.0, 40.0
    print("           protein carbon fat    ")
    print("benchmark :", protein, carbon, fat)
    hill_climbing(N, protein, carbon, fat, "oil")


if __name__ == "__main__":
    main()
