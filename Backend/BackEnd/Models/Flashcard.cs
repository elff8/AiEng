using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace BackEnd.Models
{
	public class Flashcard
	{
		public int Id { get; set; }
		public string EnglishWord { get; set; }  
		public string Transcription { get; set; }  
		public string Translation { get; set; }  

		public int? CategoryId { get; set; }
		public Category Category { get; set; }
		// Внешний ключ
		public int DifficultyLevelId { get; set; }

		// Навигационное свойство
		public DifficultyLevel DifficultyLevel { get; set; }



	}

}
//dotnet ef migrations add <MigrationName>
//dotnet ef database update 
//http://localhost:5000/api/learning/MissingWord
