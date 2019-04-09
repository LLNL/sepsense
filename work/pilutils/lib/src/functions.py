import os
from PIL import Image, ImageOps
from IPython.display import Image as IPyImage
import matplotlib.pyplot as plt
import numpy as np
try:
    import imageio
except ImportError:
    print('Warning: "imageio" library not present - GIF functions will not work')

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (220,220,220)
GRAY2 = (231,230,230)

def hstitch(images, spacing=2, border=2, spacing_color=BLACK, border_color=BLACK):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)+spacing*(len(widths)-1)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height), color=spacing_color)

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0] + spacing

    if border > 0:
        new_im = ImageOps.expand(new_im, border=border, fill=border_color)

    return new_im

def vstitch(images, spacing=2, border=2, spacing_color=BLACK, border_color=BLACK):
    widths, heights = zip(*(i.size for i in images))

    total_height = sum(heights)+spacing*(len(heights)-1)
    max_width = max(widths)

    new_im = Image.new('RGB', (max_width, total_height), color=spacing_color)

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1] + spacing

    if border > 0:
        new_im = ImageOps.expand(new_im, border=border, fill=border_color)

    return new_im

def jumble(img_list, num, jumble_size=(256, 256), imborder=2, border_color=BLACK):
    canvas = Image.new('RGB', jumble_size, color=WHITE)

    img_list = img_list[:num]

    jw, jh = jumble_size

    # Border images    
    if imborder > 0:
        new_img_list = []
        for img in img_list:
            new_img = ImageOps.expand(img, border=imborder)
            new_img_list.append(new_img)
        img_list = new_img_list

    # Pick random coords for chips
    for img in img_list:
        w, h = img.size
        mw = jw - w
        mh = jh - h
        nw = np.random.randint(0, mw)
        nh = np.random.randint(0, mh)
        canvas.paste(img, (nw, nh))

    return canvas
        

def mosaic(img_list, rows, cols, clip=True, spacing=0, imborder=1, totborder=1, spacing_color=BLACK, border_color=BLACK):
    if rows*cols < len(img_list) and clip:
        img_list = img_list[:rows*cols]
    elif rows*cols > len(img_list):
        raise Exception(
            '# images in list is less than width x height = {} x {} = {} < {}'.format(
            rows, cols, rows*cols, len(img_list)))

    # Reshape 
    img_mat = np.array(img_list, dtype=object).reshape(rows, cols).tolist()

    # Border images
    if imborder > 0:
        new_img_mat = []
        for img_row in img_mat:
            new_row = []
            for img in img_row:
                img = ImageOps.expand(img, border=imborder)
                new_row.append(img)
            new_img_mat.append(new_row)
        img_mat = new_img_mat

    # First create rows
    row_list = [hstitch(img_row, spacing=spacing, border=0, spacing_color=spacing_color, border_color=border_color) for img_row in img_mat]
    
    m = vstitch(row_list, spacing=spacing, border=0, spacing_color=spacing_color, border_color=border_color)

    if totborder > 0:
        m = ImageOps.expand(m, border=totborder)

    return m 

def dstitch(images, spacing=2, border=2, spacing_color=BLACK, border_color=BLACK, 
    xscale=6, yscale=4,
    bg_color=WHITE):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)//xscale+spacing*(len(widths)-1)+widths[0]
    total_height = sum(heights)//yscale+spacing*(len(heights)-1)+heights[0]

    new_im = Image.new('RGB', (total_width, total_height), color=bg_color)

    x_offset, y_offset = 0, 0
    for im in images:
        im = ImageOps.expand(im, border=border)
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]//xscale + spacing
        y_offset += im.size[1]//yscale + spacing

    return new_im

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D np array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a np 3D array of RGBA values
    Borrowed from: http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    """
    # draw the renderer
    fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    Borrowed from: http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    """
    # put the figure pixmap into a np array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

# Create and display a gif from a list of PIL Image objects
def pil2gif(file_name, img_list, file_dir='gif', fps=10, show=True):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = os.path.join(file_dir, file_name)
    img_arr = np.array([np.array(img) for img in img_list])
    imageio.mimwrite(file_path, img_arr, fps=fps)
    if show:
        with open(file_path,'rb') as file:
            display(IPyImage(file.read()))

# Create and display a gif from a list of matplotlib.pyplot.figure objects
def mpl2gif(file_name, fig_list, file_dir='gif', fps=10, show=True):
    img_list = [fig2img(fig) for fig in fig_list]
    pil2gif(file_name, img_list, file_dir=file_dir, fps=fps, show=show)


def lmosaic(img_list, label_list, num_rows, num_cols, plot_size=3, fontsize=12, spacing=0.02, border=3, box_color='0.8', box_lw=2, box_offset=4, box_padding=0.3, box_type='square', box_alpha=1.0, fontfamily='monospace'):
    row_size = num_rows*plot_size
    col_size = num_cols*plot_size
    fig, ax_mat = plt.subplots(nrows=num_cols, ncols=num_rows, figsize=(row_size, col_size))
    if num_rows == 1:
        ax_mat = [ax_mat]
    if num_cols == 1:
        ax_mat = [ax_mat]
    fig.subplots_adjust(wspace=spacing, hspace=spacing, left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.patch.set_facecolor('black')
    i = 0
    for ax_list in ax_mat:
        for ax in ax_list:
            ax.imshow(img_list[i])
            ax.set_xticks([]) 
            ax.set_yticks([]) 
            if label_list[i] is not None:
                ax.text(box_offset, box_offset, label_list[i], size=fontsize, va="top", ha="left", fontfamily=fontfamily,
                    bbox=dict(boxstyle="{},pad={}".format(box_type, box_padding), fc=box_color, alpha=box_alpha, lw=box_lw))
            i += 1
    plt.close()
    img = fig2img(fig)

    if border > 0:
        img = ImageOps.expand(img, border=border, fill=BLACK)

    return img

def lmosaic2(img_list, label_list2, color_list2, num_rows, num_cols, plot_size=3, fontsize=12, spacing=0.02, border=3, box_lw=2, box_offset=4, box_padding=0.3, box_type='square', box_alpha=1.0, fontfamily='monospace'):
    row_size = num_rows*plot_size
    col_size = num_cols*plot_size
    fig, ax_mat = plt.subplots(nrows=num_cols, ncols=num_rows, figsize=(row_size, col_size))
    if num_rows == 1 or num_cols == 1:
        ax_mat = [ax_mat]
    fig.subplots_adjust(wspace=spacing, hspace=spacing, left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.patch.set_facecolor('black')
    i = 0
    for ax_list in ax_mat:
        for ax in ax_list:
            ax.imshow(img_list[i])
            ax.set_xticks([]) 
            ax.set_yticks([]) 
            if label_list2[0][i] is not None:
                ax.text(box_offset, box_offset, label_list2[0][i], size=fontsize, va="top", ha="left", fontfamily=fontfamily,
                    bbox=dict(boxstyle="{},pad={}".format(box_type, box_padding), fc=color_list2[0][i], alpha=box_alpha, lw=box_lw))
                ax.text(img_list[i].size[0] - box_offset, box_offset, label_list2[1][i], size=fontsize, va="top", ha="right", fontfamily=fontfamily,
                    bbox=dict(boxstyle="{},pad={}".format(box_type, box_padding), fc=color_list2[1][i], alpha=box_alpha, lw=box_lw))
            i += 1
    plt.close()
    img = fig2img(fig)

    if border > 0:
        img = ImageOps.expand(img, border=border, fill=BLACK)

    return img

def lmosaic3(img_list, label_list3, color_list3, num_rows, num_cols, plot_size=3, fontsize=12, spacing=0.02, border=3, box_lw=2, box_offset=4, box_padding=0.3, box_type='square', box_alpha=1.0, fontfamily='monospace'):
    row_size = num_rows*plot_size
    col_size = num_cols*plot_size
    fig, ax_mat = plt.subplots(nrows=num_cols, ncols=num_rows, figsize=(row_size, col_size))
    if num_rows == 1 or num_cols == 1:
        ax_mat = [ax_mat]
    fig.subplots_adjust(wspace=spacing, hspace=spacing, left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.patch.set_facecolor('black')
    i = 0
    for ax_list in ax_mat:
        for ax in ax_list:
            ax.imshow(img_list[i])
            ax.set_xticks([]) 
            ax.set_yticks([]) 
            if label_list3[0][i] is not None:
                ax.text(box_offset, box_offset, label_list3[0][i], size=fontsize, va="top", ha="left", fontfamily=fontfamily,
                    bbox=dict(boxstyle="{},pad={}".format(box_type, box_padding), fc=color_list3[0][i], alpha=box_alpha, lw=box_lw))
                ax.text(img_list[i].size[0]//2, box_offset, label_list3[1][i], size=fontsize, va="top", ha="center", fontfamily=fontfamily,
                    bbox=dict(boxstyle="{},pad={}".format(box_type, box_padding), fc=color_list3[1][i], alpha=box_alpha, lw=box_lw))
                ax.text(img_list[i].size[0] - box_offset, box_offset, label_list3[2][i], size=fontsize, va="top", ha="right", fontfamily=fontfamily,
                    bbox=dict(boxstyle="{},pad={}".format(box_type, box_padding), fc=color_list3[2][i], alpha=box_alpha, lw=box_lw))
            i += 1
    plt.close()
    img = fig2img(fig)

    if border > 0:
        img = ImageOps.expand(img, border=border, fill=BLACK)

    return img
