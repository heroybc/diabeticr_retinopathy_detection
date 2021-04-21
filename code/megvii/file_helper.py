import os
import shutil


def gather_files(root_dir, keep_suffixs=['jpg', 'jpeg','png']):

    file_counts = {}
    for suffix in keep_suffixs:
        file_counts[suffix] = 0

    n_files = 0
    all_files = []

    for parent, dirnames, filenames in os.walk(root_dir, followlinks=False):
        for filename in filenames:

            for suffix in keep_suffixs:
                if filename.endswith(suffix):
                    file_counts[suffix] += 1

                    file_path = os.path.join(parent, filename)
                    all_files.append(file_path)

                    n_files += 1

                    break

    print('n_files = {}'.format(n_files))
    for suffix in keep_suffixs:
        print('n_files for ', suffix, ':', file_counts[suffix])

    return all_files


def filter_paths(pathes, base_names=['']):
    res = []
    for path in pathes:
        basename = os.path.basename(path)
        keep = False
        for name_tmp in base_names:
            if name_tmp == basename:
                keep = True
                break
        if keep:
            res.append(path)
    return res


def copy_imgs(from_dir, to_dir):  # 从from_dir拷贝文件至to_dir
    i_img = 0  ## 记录当前有多少张图片
    for parent, dirnames, filenames in os.walk(from_dir, followlinks=False):  # 递归遍历root_dir这个文件夹下面的所有文件和文件夹
        for from_img_name in filenames:  # 对每一个文件
            from_img_path = os.path.join(parent, from_img_name)  # 拿到原始文件路径
            suffix = os.path.splitext(from_img_path)[-1]  # 获取文件后缀
            to_img_name = str(i_img).zfill(6) + suffix  # 根据当前文件夹数量构造新的文件名，前面补0，总共6位， 如 000001.jpg, 002345.jpg
            shutil.copy(img_path, os.path.join(to_dir, to_img_name))  # 将文件拷贝至目标文件夹，并重新命名
            i_img += 1


def check_and_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def main():

    root_dir = '/data/collect-19080704004-data/19080704004_2019-09-28_09280929-chouzhen'

    gather_files(root_dir)

    pass


if __name__ == '__main__':
    main()
