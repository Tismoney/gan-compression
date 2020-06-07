import os
import subprocess


def upload_file_from_gdrive(file_id, outfile):
    path_dir = '/'.join(outfile.split('/')[:-1])
    if not os.path.exists(path_dir):
        print(f'Make dir {path_dir}')
        os.makedirs(path_dir)
    upload_cmd = (
        "wget --load-cookies /tmp/cookies.txt"
        " \"https://docs.google.com/uc?export=download&confirm=$("
        " wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies"
        f" --no-check-certificate 'https://docs.google.com/uc?export=download&id={file_id}'"
        f" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id={file_id}\" "
        f" -O {outfile} && rm -rf /tmp/cookies.txt"
    )
    subprocess.check_call(upload_cmd, shell=True)

    
if __name__ == '__main__':
    #download Teacher Net
    upload_file_from_gdrive(
        file_id='1HGEr-4cJf5rv2zgiV_8HXV_xXtdtZB01',  
        outfile='pretrained/fomm/full/latest_net_G.pth' 
    )
    
    #download Student Net
    upload_file_from_gdrive(
        file_id='19SJlK_7mngWZyPk6dhgG_ZUPdVwp-aNI',  
        outfile='pretrained/fomm/full/latest_net_G_distilled.pth' 
    )
    