	using System;
	using System.Collections.Generic;
	using System.Linq;
	using System.Threading.Tasks;

	namespace BackEnd.Models
	{
		public class MissingWord
		{
			public int Id { get; set; }
			public string SentenceWithBlank { get; set; }  // Предложение с пропущенным словом
			public string CorrectWord { get; set; }  // Правильное слово
			public int? CategoryId { get; set; }  // Категория (глагол, существительное, артикль, предлог и т.д.)
			public Category Category { get; set; }
			public string Translation { get; set; }  // Перевод предложения

			// Внешний ключ
			public int DifficultyLevelId { get; set; }

			// Навигационное свойство
			public DifficultyLevel DifficultyLevel { get; set; }

		}
	

	}
