using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace BackEnd.Models
{
	public class Test
	{
		public int Id { get; set; }
		public string Title { get; set; }
		public string Description { get; set; }

		public int TestCategoryId { get; set; }
		public TestCategory TestCategory { get; set; }

		public int DifficultyLevelId { get; set; }
		public DifficultyLevel DifficultyLevel { get; set; }

		public ICollection<Question> Questions { get; set; }
	}



}
