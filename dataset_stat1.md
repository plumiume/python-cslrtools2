# 「欠損率」をここで定義

$$
\begin{align}
    m = E_t [ \lor^V \lor^C isNaN(X_{t,v,c}) ]

    X^{All} = [ X^{Pose} X^{L-Hand} H^{R-Hand} ]
    X^{pose}
    X^{Hands} = [ X^{L-Hand} X^{R-Hand}]
    X^{L-Hand}
    X^{R-Hand}
\end{align}
$$

# 指示

idx~0 のアイテム / すべてのアイテム を T 軸で連結したアイテム$X^{Kind}$ ( $R^{T_{all}, V, C}$ ) 
を用いて「欠損率」を計算

zarrフォルダは以下の場所にあります

zarrの構造は以下の場所を参照してください
`src/cslrtools2/sldataset/dataset`

１つ前の形式のフォルダは以下の場所にあります

このファイルへの構造は提供されていません