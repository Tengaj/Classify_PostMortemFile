from file_io import classify_folder, classify_cav_folder, classify_year_folder, run_full_analysis
from classification import classify_file
from utils import CONFIG

def main():
    exec_config = CONFIG.get("execution", {})
    years_int = exec_config.get("years", [2020, 2021, 2022, 2024, 2023, 2025])
    years_str = [str(y) for y in years_int] 
    
    combine = exec_config.get("combine_csvs", False)

    input_path = exec_config.get("input_path", "../../LLRF")
    output_path = exec_config.get("output_path", "../resultat/")
    
    run_full_analysis(
        base_path=input_path,
        output_path=output_path,
        years=years_str,
        combine_all=combine
    )

if __name__ == "__main__": 

    

    main()