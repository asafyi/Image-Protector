


interface Props{
    children: string;
    
    class_name:string;
    onClick: ()=> void;
    
}

const submitButton = ({children, class_name,onClick}:Props) =>{
    return(
        

        <button className={class_name} onClick={onClick}>{children} 
        {/* <div className='spinner'></div> */}
        </button>
    )    
}
export default submitButton