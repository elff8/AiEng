using System;
using System.Collections.Generic;
namespace BackEnd.Models
{
	public class Category
	{
		public int Id { get; set; }
		public string Name { get; set; } // Название категории, например, "глагол", "существительное"
		public ICollection<Flashcard> Flashcards { get; set; }
		public ICollection<MissingWord> MissingWords { get; set; }
		public ICollection<Test> Tests { get; set; }

	}

}

