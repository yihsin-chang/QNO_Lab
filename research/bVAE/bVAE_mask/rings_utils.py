import numpy as np
import matplotlib.pyplot as plt

"""
valid_rings：判斷是否為有效的環
equiv_rings：計算等效環數
plot_rings：畫還
rings_debye：使用Debye theory計算rings對應的feild跟intensity
"""

def valid_rings(rings):
    '''
    判斷是否為有效的環
    '''
    # 環不可為空
    if len(rings) == 0:
        return False
    
    # 元素是否不是0就是1
    for element in rings:
        if (element != 0) and (element != 1):
            return False
    return True

def equiv_rings(rings):
    '''
    計算等效環數
    '''
    if not valid_rings(rings):
        return -1
    
    cnt = 1    # 不管怎麼樣都至少有一個環
    for i in range(1, len(rings)):
        if rings[i-1] != rings[i]:    # 跟前面的環不同，就構成等效環，環數就要+1
            cnt += 1
    return cnt

def plot_rings(rings, add_text=True, show=False, save=True, path="rings.png"):
    '''
    畫環
    參數
    - add_text: 要不要加實際環數、等效環數的說明文字（預設：要）
    '''
    if not valid_rings(rings):
        return -1

    # 環數
    rings_num_actual = len(rings)
    rings_num_equiv  = equiv_rings(rings)

    # 設定顏色（matplotlib color接受的RGB要是0~1的實數）
    colors = np.array(
        [(195, 230, 255),    # 0: 藍色
        (204, 153, 255)],    # 1: 紫色
        # (204, 0, 0)],
        dtype=float
    )
    colors /= 255.0

    # 建立figure跟axes
    fig = plt.figure(figsize=(10,10), dpi=100)
    ax = fig.add_subplot()
    ax.set_aspect('equal')    # 確保圖的長寬比是1:1，才能保證Circle是正圓形

    # 設定axes範圍
    ax.set_xlim(left=-1.1, right=2.2)
    ax.set_ylim(bottom=-1.1, top=1.1)
    
    # 畫同心圓，從最外層開始畫（最大半徑為1才能在axes的範圍內）
    for i in range(rings_num_actual-1, -1, -1):
        circle = plt.Circle(
            xy=(0, 0),
            radius=i/rings_num_actual,    # 正規化，讓最大的半徑為1
            color=colors[int(rings[i])],
            # ec="black"
        )
        ax.add_artist(circle)
    
    # 顯示文字
    if add_text:
        text = f"• Actual Rings: {rings_num_actual}\n" \
               f"• Equiv Rings:  {rings_num_equiv}"
        plt.text(
            x = 0.9, y = 0.9,
            s = text,       # 解釋文字
            fontsize=14,    # 文字大小
            ha="left",      # 水平對齊方式
            va="top"        # 垂直對齊方式
        )
    
    # 移除軸刻度
    ax.axis('off')
    
    # 顯示圖形
    if save:
        plt.savefig(fname=path, transparent=True, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()

def rings_debye(rings):
    pass

if __name__ == "__main__":
    rings = np.array([1,0,0,1,0,0,1,0,1,
                     1,0,0,1,0,0,1,0,1,
                     1,0,0,1,0,0,1,0,1,
                     1,0,0,1,0,0,1,0,1,
                     1,0,0,1,0,0,1,0,1,
                     1,0,0,1,0,0,1,0,1,
                     1,0,0,1,0,0,1,0,1,
                     1,0,0,1,0,0,1,0,1,
                     1,0,0,1,0,0,1,0,1,
                     1,0,0,1,0,0,1,0,1,
                     1,0,0,1,0,0,1,0,1,
                     1,0,0,1,0,0,1,0,1,
                     1,0,0,1,0,0,1,0,1,
                     1,0,0,1,0,0,1,0,1,
                     1,0,0,1,0,0,1,0,1,
                     1,0,0,1,0,0,1,0,1,
                     1,0,0,1,0,0,1,0,1,
                     1,0,0,1,0,0,1,0,1,
                     1,0,0,1,0,0,1,0,1,
                     1,0,0,1,0,0,1,0,1,], dtype=float)
    print(f"valid: {valid_rings(rings)}")
    print(f"equiv: {equiv_rings(rings)}")
    plot_rings(rings)