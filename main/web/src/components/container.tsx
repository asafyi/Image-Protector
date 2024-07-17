import type { FC } from 'react'
import { useCallback, useState } from 'react'

import { FileList } from './FileList'
import { TargetBox } from './TargetBox'

import FileUploader from './FileUploader'

export const Container: FC = () => {
  const [droppedFiles, setDroppedFiles] = useState<File[]>([])

  const handleFileDrop = useCallback(
    (item: { files: any[] }) => {
      if (item) {
        const files = item.files
        setDroppedFiles(files)
      }
    },
    [setDroppedFiles],
  )

  

  const handleFileUpload = (files: FileList | null) => {
    if (files) {
      const uploadedFiles = Array.from(files) as any[];
      setDroppedFiles(uploadedFiles);
    }

  };

  const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
  


  const handleSubmit = async (event: { currentTarget: any }) => {
    var obj =event.currentTarget;
 
    obj.innerHTML="<div class='loader'></div>"

    //await sleep(3000);

    if(droppedFiles.length==0){
      alert("No files were uploaded. Please upload files to protect");
    }
      else if( droppedFiles.length>5){
        alert("Please upload up to 5 files to protect");
      }

      else{
    
    const formData = new FormData();
    
    for (let i=0;i<droppedFiles.length;i++){
      formData.append("files",droppedFiles[i])

    }

    
    var oReq = new XMLHttpRequest();
    oReq.responseType = "blob";
    await oReq.open("POST", "upload", true); 
    
  


    oReq.onload = function() {//oEvent

      

       if (oReq.status == 200 && droppedFiles.length>0&& droppedFiles.length<6) {
  
          var blob = new Blob([oReq.response]);
          var g_url = window.URL.createObjectURL(blob);
          // create a new anchor element with a download attribute and a href attribute set to the URL
        var link = document.createElement('a');
        link.href = g_url;
        if (droppedFiles.length == 1){
          link.download = droppedFiles[0].name;
        } else {
          link.download = "all.zip";   
        }
      
        document.body.appendChild(link);
        // click the anchor element
        link.click();
        // remove the anchor element from the document body
        document.body.removeChild(link);
        
        obj.innerHTML="create protected image!";
      } else {
      
          alert("Error occurred when trying to upload your file.");
        
      }
  }; 

  await oReq.send(formData);

}


  };
  




const handleDelete=async (event: { currentTarget: any })=>{
  
  var obj =event.currentTarget;
    obj.classList.add('ui', 'loading', 'button');
    //while(droppedFiles.length>0){
      setDroppedFiles([]);//empty file list
      await sleep(200);
    //}
    obj.classList.remove('ui', 'loading', 'button');
};

  return (
    <>
    
    <TargetBox  onDrop={handleFileDrop}>{<FileUploader onFileUpload={handleFileUpload} />}</TargetBox> 
    
    
    
    <div className='buttons'>

    <button  className='submitButton generalButtons' onClick={handleSubmit}>create protected image! </button>
    <button className=" deleteButton generalButtons" onClick= {handleDelete}>remove uploaded files</button>
    </div>

    <div className='centerText'>

      <FileList id='fileListId' name='fileListId' files={droppedFiles} />
      
    </div>
      

      
    </>
  )
}
