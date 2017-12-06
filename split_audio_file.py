from glob import glob
period = 4
with open('split_command.txt', 'w') as f:
    for filetype in ['.wav', '.mp3']:
        for filename in glob('*{filetype}'.format(filetype=filetype)):
            for i in range(10):
                command = 'sox {fn}{filetype} {fn}_{i}{filetype} --show-progress trim {start} {end}\n'.format(
                    fn=filename.split(filetype)[0],
                    filetype=filetype,
                    i=i + 1,
                    start=period * i,
                    end=period
                )
                f.write(command)

