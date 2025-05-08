using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace BackEnd.Models
{
	public class DifficultyLevel
	{
		public int Id { get; set; }
		public string Name { get; set; }
		public string Description { get; set; }  // Описание уровня (опционально)
		public ICollection<Flashcard> Flashcards { get; set; }
		public ICollection<MissingWord> MissingWords { get; set; }
		public ICollection<Test> Tests { get; set; }

	}

}
//dotnet ef migrations add <MigrationName>
//dotnet ef database update 