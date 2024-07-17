import type {  FC } from 'react'//CSSProperties
import type { DropTargetMonitor } from 'react-dnd'
import { useDrop } from 'react-dnd'
import { NativeTypes } from 'react-dnd-html5-backend'
//import { Children, useCallback, useState } from 'react'
import React from 'react'






export interface TargetBoxProps {
  onDrop: (item: { files: any[] }) => void
  children?: React.ReactNode|React.ReactNode[];
}



export const TargetBox: FC<TargetBoxProps> = ({onDrop,children}: TargetBoxProps) => {
  const childrenArray = React.Children.toArray(children);
  const [{ canDrop, isOver }, drop] = useDrop(
    () => ({
      accept: [NativeTypes.FILE],
      drop(item: { files: any[] }) {
        if (onDrop) {
          onDrop(item)
        }
      },
      // canDrop(item: any) {
      //   //console.log('canDrop', item.files, item.items)
      //   return true
      // },
 
      // hover(item: any) {
      //   console.log('hover', item.files, item.items)
      // },
      collect: (monitor: DropTargetMonitor) => {
        
        //const item = monitor.getItem() as any
        // if (item) {
        //   console.log('collect', item.files, item.items)
        // }

        return {
          isOver: monitor.isOver(),
          canDrop: monitor.canDrop(),
        }
      },
    }),
   
  )

  const isActive = canDrop && isOver

  return (
    <>

    <div ref={drop} className='white boxStyle' >
      
      {isActive ?  'Release to drop' : 'Drag files here'}
      
    {childrenArray}  
    </div>
    </>
  )
}
