<!DOCTYPE html>
<html>

<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <h1>🥈2nd place solution HSE Sber RecSys Hack🥈</h1>
  <link rel="stylesheet" href="https://stackedit.io/style.css" />
</head>

<body class="stackedit">
  <h1>Classic RecSys challenge</h1>
  <h3>As we analyzed the data, we saw an imbalance in the ratings, shown below</h3>
  <img  src="https://github.com/SoberSinceToday/HSE-Sber-RecSys-Hack/blob/main/git_utils/pic1.jpg"/>
  <img  src="https://github.com/SoberSinceToday/HSE-Sber-RecSys-Hack/blob/main/git_utils/pic2.jpg"/>
  <h3>After preprocessing the data, which was to create a less sparse matrix, I trained LightGCN, which showed the following results:</h3>
  <img src="https://github.com/SoberSinceToday/HSE-Sber-RecSys-Hack/blob/main/git_utils/pic3.png"/>

  <h2>Repository directory:</h2>

  <body>
    <ul style="font-size: 34px;">
        <li>🤖model.py, lightgcn.yaml - model class and it's params🤖</li>
        <li>🧩train_predict.py - decision file containing data processing and model prediction🧩</li>
        <li>🔧utils_for_readme, requirements.txt - auxiliary files for README and requirements🔧</li>
    </ul>
</body>
</body>
</html>
