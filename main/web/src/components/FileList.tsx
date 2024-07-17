import type { FC } from 'react'
import { useMemo } from 'react'
export interface FileListProps {
  files: File[],
  id:string,
  name:string,
}

function list(files: File[]) {
  const label = (file: File) =>
    `${file.name}`//`'${file.name}' of size '${file.size}' and type '${file.type}'`
  return files.map((file) => <li key={file.name}>{label(file)}</li>)
}

export const FileList: FC<FileListProps> = ({ files }) => {

  const fileList = useMemo(() => list(files), [files])
  return <div className='black'>{fileList}</div>
}
